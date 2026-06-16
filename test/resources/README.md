# Test resources

## `speech.wav`

A ~11 second 16 kHz mono PCM clip of John F. Kennedy's 1961 inaugural address
("And so, my fellow Americans...").

- **Provenance:** widely distributed as `samples/jfk.wav` in the
  [whisper.cpp](https://github.com/ggerganov/whisper.cpp) project.
- **License:** public domain — a speech by a U.S. federal government official in the
  course of official duties is not subject to copyright (17 U.S.C. § 105).

Used to verify that backends detect real speech offline (no network required).
