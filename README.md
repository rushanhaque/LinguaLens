# LinguaLens

LinguaLens helps you learn words by pointing your camera at everyday objects. It uses AI-powered object detection, with generative AI assistance during development. It recognises things like a cup, chair or phone and shows their names in French, Spanish, German, Japanese or Italian.

You can tap a word to hear it, revisit the objects you have discovered in the vocabulary panel, or use quiz mode to hide labels until you tap them. There is also a snapshot button to save the camera view with its detection boxes.

## How it works

The app uses HTML, CSS and JavaScript. TensorFlow.js runs a pretrained COCO-SSD object detection model in the browser. Camera frames are processed on your device, and translations come from the entries in `dictionary.js`.

The detection code filters overlapping boxes, checks object proportions and waits for repeated detections before showing a label. This helps reduce flickering and some incorrect guesses. Pronunciation uses the browser's speech synthesis, and discovered vocabulary is saved in local storage.

## Run it locally

Clone the repository and start a local server with Python 3:

```sh
git clone https://github.com/rushanhaque/LinguaLens.git
cd LinguaLens
python -m http.server 8000 --bind 127.0.0.1
```

Open [localhost:8000](http://localhost:8000) and allow camera access. On Windows, you can use `py` instead of `python`. There is no build step or API key to set up.

Use an internet connection for the first load so the browser can download the model and libraries. A hosted copy needs HTTPS for camera access.

## Where Python fits

Python provides a separate vocabulary export tool in `tools/export_vocab.py`. It reads the same dictionary the app uses and creates a CSV study sheet with English words and their translations. You can open the sheet in Excel or print it for revision.

The tool uses **json5** to read the dictionary's JavaScript object as data and **pandas** to organise, sort and export the words. It does not execute JavaScript or change the app's files. Python's built-in `argparse` and `pathlib` handle command options and file paths.

With Python 3.10 or newer, run:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on macOS/Linux. Then run:

```sh
python -m pip install -r requirements.txt
python tools/export_vocab.py
```

This saves all five languages to `exports/vocabulary.csv`. To make a Japanese study sheet:

```sh
python tools/export_vocab.py --language ja --output exports/japanese.csv
```

The language options are `fr`, `es`, `de`, `ja` and `it`. Choose a new output filename if one already exists; the tool will not overwrite it. Generated sheets stay out of Git.

To check the export tool, run `python -m unittest discover -s tools -p "test_*.py"`. The tests cover all 80 words, accented and Japanese text, single-language exports, invalid input and overwrite protection.

Python can also serve the website locally with `http.server`, as shown above. The website itself needs no Python packages or backend, and detection still runs in the browser.

## Generative AI during development

Generative AI was used as a coding assistant. Codex helped write the Python export tool and this README, and run checks on the exported vocabulary. The repository's later commits also credit Claude for work on the app rebuild, vocabulary expansion and interface changes. Those changes remain in Git history after the return to the February 24 app version.

The running app does not call a generative AI service. It uses a pretrained object detector and a fixed translation dictionary, so it does not generate translations or answers on demand.

## Main files

- `index.html` and `styles.css`: page layout and appearance.
- `app.js`: camera setup, labels, pronunciation, quiz and vocabulary.
- `detector.js`: filtering and stabilising object detections.
- `dictionary.js`: translations and object size hints.
- `sw.js` and `manifest.json`: caching and web app installation support.
- `tools/export_vocab.py` and `requirements.txt`: the optional Python study sheet tool and its dependencies.

Detection can make mistakes, especially in poor lighting or when objects look alike. It only recognises the categories supported by the model. Available pronunciation voices depend on your browser and device.

Built by Rushan Haque.
