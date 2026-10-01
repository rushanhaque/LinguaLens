# LinguaLens

LinguaLens helps you learn words by pointing your camera at everyday objects. It recognises things like a cup, chair or phone and shows their names in French, Spanish, German, Japanese or Italian.

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

Python can serve the project locally using its built-in `http.server` module. This needs no extra Python packages. The current version has no Python backend; object detection runs through TensorFlow.js in the browser.

In the later development history, Python was also used to create a PDF pitch deck with **ReportLab**. That script used ReportLab's drawing, font and image utilities, along with Python's built-in `os` module. The [pitch-deck script](https://github.com/rushanhaque/LinguaLens/blob/46b409d79ecd08dc0d8a4e2cd7331270918af394/tools/make-deck.py) is preserved in Git history, but is not included in this restored February 24 version. ReportLab is not needed to run the app.

## Generative AI during development

Generative AI was used as a coding assistant during development. The repository's later commits credit Claude for work on the app rebuild, vocabulary expansion and interface changes. Those changes remain in Git history after the return to the February 24 version. This README was also prepared with Codex after checking the source code.

The running app does not call a generative AI service. It uses a pretrained object detector and a fixed translation dictionary, so it does not generate translations or answers on demand.

## Main files

- `index.html` and `styles.css`: page layout and appearance.
- `app.js`: camera setup, labels, pronunciation, quiz and vocabulary.
- `detector.js`: filtering and stabilising object detections.
- `dictionary.js`: translations and object size hints.
- `sw.js` and `manifest.json`: caching and web app installation support.

Detection can make mistakes, especially in poor lighting or when objects look alike. It only recognises the categories supported by the model. Available pronunciation voices depend on your browser and device.

Built by Rushan Haque.
