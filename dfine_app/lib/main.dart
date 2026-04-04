import 'dart:typed_data';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

void main() {
  runApp(const DFineApp());
}

class DFineApp extends StatelessWidget {
  const DFineApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'D-FINE VisDrone Detector',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1565C0),
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const DetectorPage(),
    );
  }
}

// ── Data model ──────────────────────────────────────────────────────────────

class Detection {
  final String label;
  final double score;
  final double x1, y1, x2, y2;

  const Detection({
    required this.label,
    required this.score,
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
  });

  factory Detection.fromJson(Map<String, dynamic> j) {
    final box = j['box'] as Map<String, dynamic>;
    return Detection(
      label: j['label'] as String,
      score: (j['score'] as num).toDouble(),
      x1: (box['x1'] as num).toDouble(),
      y1: (box['y1'] as num).toDouble(),
      x2: (box['x2'] as num).toDouble(),
      y2: (box['y2'] as num).toDouble(),
    );
  }
}

// ── Colour map for 10 VisDrone classes ──────────────────────────────────────

const _classColors = <String, Color>{
  'pedestrian':      Color(0xFFE53935),
  'people':          Color(0xFFFF7043),
  'bicycle':         Color(0xFFFDD835),
  'car':             Color(0xFF43A047),
  'van':             Color(0xFF00ACC1),
  'truck':           Color(0xFF1E88E5),
  'tricycle':        Color(0xFF8E24AA),
  'awning-tricycle': Color(0xFFD81B60),
  'bus':             Color(0xFFFF8F00),
  'motor':           Color(0xFF6D4C41),
};

Color _colorFor(String label) =>
    _classColors[label] ?? const Color(0xFFFFFFFF);

// ── Box painter ──────────────────────────────────────────────────────────────

class _BoxPainter extends CustomPainter {
  final List<Detection> detections;
  final double imgW, imgH; // original image dimensions
  final ui_Size canvasSize;

  _BoxPainter(this.detections, this.imgW, this.imgH, this.canvasSize);

  @override
  void paint(Canvas canvas, Size size) {
    // Match BoxFit.contain: scale uniformly, center within widget
    final scale   = (size.width / imgW).clamp(0.0, size.height / imgH);
    final offsetX = (size.width  - imgW * scale) / 2;
    final offsetY = (size.height - imgH * scale) / 2;

    for (final d in detections) {
      final color = _colorFor(d.label);
      final paint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0;

      final rect = Rect.fromLTRB(
        d.x1 * scale + offsetX, d.y1 * scale + offsetY,
        d.x2 * scale + offsetX, d.y2 * scale + offsetY,
      );
      canvas.drawRect(rect, paint);

      final label = '${d.label} ${(d.score * 100).toStringAsFixed(0)}%';
      final tp = TextPainter(
        text: TextSpan(
          text: label,
          style: TextStyle(
            color: Colors.white,
            fontSize: 11,
            background: Paint()..color = color.withAlpha(180),
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, Offset(rect.left + 2, rect.top + 2));
    }
  }

  @override
  bool shouldRepaint(_BoxPainter old) =>
      old.detections != detections || old.canvasSize != canvasSize;
}

// ignore: library_private_types_in_public_api
typedef ui_Size = Size;

// ── Main page ────────────────────────────────────────────────────────────────

class DetectorPage extends StatefulWidget {
  const DetectorPage({super.key});

  @override
  State<DetectorPage> createState() => _DetectorPageState();
}

class _DetectorPageState extends State<DetectorPage> {
  // ---- state ----
  Uint8List? _imageBytes;
  String? _imageUrl;   // base64 data URL for reliable web rendering
  double _imgW = 1, _imgH = 1;
  List<Detection> _detections = [];
  bool _loading = false;
  String? _error;
  String _serverUrl    = 'http://localhost:8000';
  String _selectedModel = 'yolov8';
  double _threshold     = 0.3;

  final _picker = ImagePicker();
  final _urlController = TextEditingController(text: 'http://localhost:8000');

  // ---- helpers ----

  Future<void> _pickAndDetect(ImageSource source) async {
    try {
      final xfile = await _picker.pickImage(source: source, imageQuality: 100);
      if (xfile == null) return;
      final bytes = await xfile.readAsBytes();
      await _runDetection(bytes, xfile.name);
    } catch (e) {
      setState(() => _error = 'Image pick failed: $e');
    }
  }

  Future<void> _runDetection(Uint8List bytes, String filename) async {
    // Detect MIME type from magic bytes
    String mime = 'image/jpeg';
    if (bytes.length > 4 &&
        bytes[0] == 0x89 && bytes[1] == 0x50 &&
        bytes[2] == 0x4E && bytes[3] == 0x47) {
      mime = 'image/png';
    }
    final dataUrl = 'data:$mime;base64,${base64Encode(bytes)}';

    setState(() {
      _loading = true;
      _error = null;
      _detections = [];
      _imageBytes = bytes;
      _imageUrl = dataUrl;
    });

    try {
      final uri = Uri.parse('$_serverUrl/detect');
      final req = http.MultipartRequest('POST', uri)
        ..fields['model']     = _selectedModel
        ..fields['threshold'] = _threshold.toStringAsFixed(2)
        ..files.add(http.MultipartFile.fromBytes(
          'file', bytes,
          filename: filename,
        ));
      final streamed = await req.send().timeout(const Duration(seconds: 30));
      final body = await streamed.stream.bytesToString();

      if (streamed.statusCode != 200) {
        setState(() { _loading = false; _error = 'Server error ${streamed.statusCode}: $body'; });
        return;
      }

      final json = jsonDecode(body) as Map<String, dynamic>;
      final dets = (json['detections'] as List)
          .map((e) => Detection.fromJson(e as Map<String, dynamic>))
          .toList();

      setState(() {
        _loading = false;
        _imgW = (json['image_width'] as num).toDouble();
        _imgH = (json['image_height'] as num).toDouble();
        _detections = dets;
      });
    } catch (e) {
      setState(() { _loading = false; _error = 'Request failed: $e'; });
    }
  }

  // ---- build ----

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('D-FINE VisDrone Detector'),
        backgroundColor: Theme.of(context).colorScheme.primaryContainer,
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: _showSettings,
            tooltip: 'Server URL',
          ),
        ],
      ),
      body: Column(
        children: [
          // ── model selector + threshold ─────────────────────────────────
          Container(
            color: Theme.of(context).colorScheme.surfaceContainerLow,
            padding: const EdgeInsets.fromLTRB(16, 6, 16, 4),
            child: Column(
              children: [
                DropdownButton<String>(
                  value: _selectedModel,
                  isExpanded: true,
                  underline: const SizedBox(),
                  items: const [
                    DropdownMenuItem(value: 'yolov8', child: Text('YOLOv8-X  (SOTA · 68M params · AP50=0.470)')),
                    DropdownMenuItem(value: 'dfine',  child: Text('D-FINE-S  (ours · 10M params · AP50=0.389)')),
                  ],
                  onChanged: (v) { if (v != null) setState(() => _selectedModel = v); },
                ),
                Row(
                  children: [
                    const Text('Threshold:', style: TextStyle(fontSize: 13)),
                    Expanded(
                      child: Slider(
                        value: _threshold,
                        min: 0.05,
                        max: 0.95,
                        divisions: 18,
                        label: _threshold.toStringAsFixed(2),
                        onChanged: (v) => setState(() => _threshold = v),
                      ),
                    ),
                    Text(_threshold.toStringAsFixed(2),
                        style: const TextStyle(fontSize: 13, fontFamily: 'monospace')),
                  ],
                ),
              ],
            ),
          ),

          // ── image + boxes ──────────────────────────────────────────────
          Expanded(
            child: _imageBytes == null
                ? const Center(
                    child: Text(
                      'Pick an image to detect objects',
                      style: TextStyle(color: Colors.white54, fontSize: 16),
                    ),
                  )
                : _loading
                    ? const Center(child: CircularProgressIndicator())
                    : LayoutBuilder(builder: (ctx, constraints) {
                        return Stack(
                          fit: StackFit.expand,
                          children: [
                            Image.memory(_imageBytes!, fit: BoxFit.contain),
                            CustomPaint(
                              painter: _BoxPainter(
                                _detections, _imgW, _imgH,
                                Size(constraints.maxWidth, constraints.maxHeight),
                              ),
                            ),
                          ],
                        );
                      }),
          ),

          // ── error banner ───────────────────────────────────────────────
          if (_error != null)
            Container(
              color: Colors.red.shade900,
              padding: const EdgeInsets.all(8),
              width: double.infinity,
              child: Text(_error!, style: const TextStyle(color: Colors.white)),
            ),

          // ── detection count ────────────────────────────────────────────
          if (_detections.isNotEmpty)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 4),
              child: Text(
                '${_detections.length} detections',
                style: const TextStyle(color: Colors.white70),
              ),
            ),

          // ── buttons ────────────────────────────────────────────────────
          Padding(
            padding: const EdgeInsets.all(12),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed: _loading ? null : () => _pickAndDetect(ImageSource.gallery),
                  icon: const Icon(Icons.photo_library),
                  label: const Text('Gallery'),
                ),
                ElevatedButton.icon(
                  onPressed: _loading ? null : () => _pickAndDetect(ImageSource.camera),
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('Camera'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  void _showSettings() {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Server URL'),
        content: TextField(
          controller: _urlController,
          decoration: const InputDecoration(
            labelText: 'e.g. http://localhost:8000',
            border: OutlineInputBorder(),
          ),
          onSubmitted: (v) {
            setState(() => _serverUrl = v.trim());
            Navigator.pop(context);
          },
        ),
        actions: [
          TextButton(
            onPressed: () {
              setState(() => _serverUrl = _urlController.text.trim());
              Navigator.pop(context);
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
  }
}
