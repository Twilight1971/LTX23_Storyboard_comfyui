# LTX23 Storyboard Splitter for ComfyUI

Custom Node zum Zerlegen eines Seedance/LTX-Storyboard-Sheets in einzelne Shot-Bilder und Textbereiche.

## Installation

Kopiere diesen Ordner nach:

```text
ComfyUI/custom_nodes/LTX23_Storyboard_comfyui
```

Starte ComfyUI neu. Der Node erscheint unter:

```text
LTX/Storyboard -> LTX Storyboard Splitter
```

## Outputs

- `panel_images`: Batch mit den einzelnen Storyboard-Bildern.
- `timing_setting_text_images`: Batch mit linker Spalte pro Shot.
- `description_text_images`: Batch mit Beschreibungsspalte pro Shot.
- `camera_text_images`: Batch mit Kamera/Bewegung-Spalte pro Shot.
- `storyboard_json`: Strukturierte Shot-Daten als JSON.
- `shot_prompts`: Ein Prompt pro Shot, geeignet zum Weiterreichen an Text-/Video-Nodes.
- `timing_csv`: CSV-Text für Tabellen oder Logging.

## OCR

Der Node funktioniert auch ohne OCR und gibt dann zuverlässig alle Crop-Batches aus. Für automatische Texterkennung:

1. Installiere Tesseract OCR für Windows.
2. Installiere optional `pytesseract` in der Python-Umgebung von ComfyUI.
3. Setze `enable_ocr` auf `true`.

Wenn OCR nicht verfügbar ist, bleiben die Textfelder im JSON leer, die Text-Bildbereiche werden aber weiterhin ausgegeben.

## Automatisches Speichern

Zusätzlich gibt es:

```text
LTX/Storyboard -> LTX Storyboard Asset Saver
LTX/Storyboard -> LTX Storyboard Video Prompt Builder
```

Verbinde die Outputs des Splitters mit diesem Saver. Er schreibt pro Durchlauf in ComfyUIs `output`-Ordner:

- `shot_01.png` bis `shot_06.png`
- optionale Text-Crops pro Spalte
- `storyboard.json`
- `shot_prompts.txt`
- `timing.csv`

Der `Video Prompt Builder` macht aus `storyboard_json` LTX-2.3-Image-to-Video-Standardprompts, einen zusammenhaengenden Master-Prompt, den Negative Prompt, `fallback_frames_per_scene`, `total_frames`, `scene_timing_json` und `frame_counts_csv`.

Jeder erkannte Scene-Prompt wird als I2V-Clip-Prompt formatiert:

```text
LTX-2.3 Image-to-Video prompt for scene X.
Use the provided image as the first frame and visual reference.
Preserve the source image identity, subject, outfit, environment, framing, aspect ratio and color palette.
Action and motion: ...
Camera movement: ...
Shot timing and framing: ...
Clip length: ...
Storyboard timecode: ...
Visual style: ...
Audio cues: ...
Avoid changing the main subject identity, composition, object count or scene layout unless the storyboard explicitly asks for it.
```

Timecodes aus dem Storyboard werden automatisch beruecksichtigt, wenn OCR sie erkennt, z. B. `0:00 - 0:02`, `0-2 SEC` oder `8-10 SEC`. Dann berechnet der Node pro Szene die echte Dauer und Frame-Anzahl. Wenn kein Timecode erkannt wird, verwendet er `seconds_per_scene` als Fallback.

## LTX 2.3 Video Workflow

Nutze [workflows/storyboard_to_ltx23_video_template.json](workflows/storyboard_to_ltx23_video_template.json) als Verdrahtungsplan:

1. `LoadImage` lädt das Storyboard.
2. `LTX Storyboard Splitter` zerlegt es in Panels und Textdaten.
3. `LTX Storyboard Video Prompt Builder` erzeugt konsistente Shot-Prompts.
4. Die `panel_images` gehen in die Image-to-Video-Strecke des offiziellen LTX-2.3-Workflows.
5. Pro Szene wird ein kurzer Clip generiert.
6. Die Clips werden mit VideoHelperSuite/`Video Combine` oder ComfyUIs `SaveVideo` hintereinander gespeichert.

Empfohlen ist der offizielle LTX-2.3 Image-to-Video Workflow aus ComfyUIs Template Library oder aus `ComfyUI-LTXVideo/example_workflows/2.3`. Ersetze dort:

- `Load Image` durch `panel_images` vom Splitter.
- positiven Prompt durch `ltx_scene_prompts` oder den jeweiligen LTX-2.3-I2V-Scene-Prompt.
- Negative Prompt durch den Output `negative_prompt`.
- Frame-Zahl pro Szene aus `frame_counts_csv` oder `scene_timing_json`. Wenn keine Timecodes erkannt werden, nutze `fallback_frames_per_scene`, z. B. 48 Frames bei 24 fps und 2 Sekunden pro Shot.

## Kompletter LTX-2.3 Render

Enthalten sind zwei praktische Workflows:

- `workflows/storyboard_to_ltx23_complete_scene_render.json`
- `workflows/storyboard_to_ltx23_complete_scene_render_rtx4060ti_16gb_safe.json`
- `workflows/storyboard_to_ltx23_complete_scene_render_rtx4060ti_16gb_fixed.json`
- `workflows/storyboard_ltx23_concat_rendered_scenes.json`

`storyboard_to_ltx23_complete_scene_render.json` basiert auf dem offiziellen Lightricks `LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full` Workflow. Vorne sind Storyboard-Load, Splitter, Prompt Builder und Scene Selector eingefuegt. Stelle im `LTX Storyboard Scene Selector` den `scene_index` auf `1`, `2`, `3` usw. und rendere die Szenenclips nacheinander. Der Scene Selector uebergibt automatisch:

- das passende Szenenbild an `LTXVPreprocess`
- den passenden LTX-2.3-I2V-Prompt an Positive Text Encode
- den Negative Prompt an Negative Text Encode
- die aus dem Timecode berechnete Framezahl an LTX Video/Audio Length

Danach kannst du die gerenderten MP4-Dateien mit `storyboard_ltx23_concat_rendered_scenes.json` zu einem finalen Video verbinden. Die Dateien werden alphabetisch sortiert, daher am besten `scene_01.mp4`, `scene_02.mp4`, `scene_03.mp4` usw. verwenden.

### RTX 4060 Ti 16GB Profil

Fuer eine RTX 4060 Ti mit 16GB VRAM nutze zuerst:

```text
workflows/storyboard_to_ltx23_complete_scene_render_rtx4060ti_16gb_safe.json
```

Dieses Profil ist konservativer eingestellt:

- `ltx-2.3-22b-dev-fp8.safetensors` statt BF16 Full Checkpoint
- `768 x 432` Latent-Aufloesung fuer Landscape
- `24 fps`
- `48` Frames fuer 2-Sekunden-Szenen
- konservativeres tiled VAE Decode Profil
- Szene-fuer-Szene Rendering statt alle Szenen gleichzeitig

Fuer Vertical/Reel 9:16 stelle `EmptyLTXVLatentVideo` auf:

```text
432 x 768
```

Wenn ComfyUI trotzdem Out-of-Memory meldet:

- Framezahl testweise auf `33` reduzieren
- Aufloesung auf `640 x 360` oder `360 x 640` reduzieren
- ComfyUI mit FP8-Modell starten und keine anderen grossen Workflows offen lassen
- Nach jedem grossen Render Cache leeren oder ComfyUI neu starten

### Bekannte LTX-2.3 Textencoder/Audio-VAE Fehler

Wenn ComfyUI bei `CLIPTextEncode` mit

```text
AttributeError: 'Linear' object has no attribute 'weight'
linear(): argument 'weight' must be Tensor, not NoneType
```

abbricht, liegt das nicht am Storyboard-Splitter. Der Fehler kommt vom LTX-2.3 Textencoder-Load. Typische Ursachen:

- ComfyUI ist nicht auf einem aktuellen LTX-2.3-kompatiblen Stand.
- `ComfyUI-LTXVideo` ist veraltet.
- `gemma_3_12B_it_fp4_mixed.safetensors` ist kaputt, unvollstaendig oder im falschen Ordner.
- Der FP4/Mixed Textencoder ist mit der aktuell installierten PyTorch/ComfyUI-Kombination inkompatibel.

Wenn vorher

```text
buffer length ... must be a multiple of element size
```

auftaucht, ist sehr wahrscheinlich ein Safetensors-Download kaputt, meistens Audio-VAE. Dann die LTX-2.3 Audio-VAE-Datei neu herunterladen.

Zum Testen nutze:

```text
workflows/storyboard_to_ltx23_complete_scene_render_rtx4060ti_16gb_fixed.json
```

Diese Version reduziert die Testlast auf `640 x 360` und `33` Frames. Sie ersetzt aber keine defekten Modelldateien.

## Variable Storyboard-Layouts

Der Splitter kann unterschiedliche GPT-Storyboard-Formate verarbeiten:

- `auto`: erkennt das Layout grob selbst.
- `seedance_wide_rows`: breites 6-Zeilen-Layout mit Bild/Description/Kamera-Spalten.
- `two_column_cards`: TikTok/Reel-Storyboard mit 2 Spalten und mehreren Karten.
- `light_table_rows`: helles Tabellenlayout mit Szene/Zeit, Referenzbild, Action und Overlay/Effekte.
- `dark_table_rows`: dunkles vertikales Film-Storyboard mit linker Szenenspalte, Bild und rechter Beschreibung.

`scene_count = 0` nutzt die Zeilenzahl aus `rows`. Fuer mehr Szenen setze `rows` oder `scene_count` auf die gewuenschte Anzahl. Bei leicht abweichenden Exporten kann `crop_inset_px` angepasst werden.
