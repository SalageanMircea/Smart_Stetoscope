# Smart_Stetoscope

Smart_Stetoscope is a PyTorch project that uses a ResNet34 to detect heartbeat anomalies from auscultation audio.

It preprocesses WAV recordings (resample to 4 kHz, 20–500 Hz band-pass, normalization) and extracts multiple 5-second segments per file.

Each segment is converted to a log-Mel spectrogram, with waveform + SpecAugment-style masking during training.

Segment predictions are combined with an attention pooling layer to output one label per recording (normal / murmur / artifact / extra).

The script trains, saves the best checkpoint, and exports evaluation outputs like a confusion matrix image.

<img width="1372" height="1183" alt="confusion_matrix" src="https://github.com/user-attachments/assets/a0195941-0b15-444e-82c8-5565132d56ad" />

<img width="492" height="239" alt="Screenshot 2026-01-06 203226" src="https://github.com/user-attachments/assets/886611fd-b431-4c5f-ac80-f46384037861" />
