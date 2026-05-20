import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps


@dataclass(frozen=True)
class Box:
    left: float
    top: float
    right: float
    bottom: float

    def pixels(self, width: int, height: int, inset: int = 0) -> Tuple[int, int, int, int]:
        l = int(round(self.left * width)) + inset
        t = int(round(self.top * height)) + inset
        r = int(round(self.right * width)) - inset
        b = int(round(self.bottom * height)) - inset
        l = max(0, min(width - 1, l))
        t = max(0, min(height - 1, t))
        r = max(l + 1, min(width, r))
        b = max(t + 1, min(height, b))
        return l, t, r, b


SEEDANCE_2_LAYOUT = {
    "left": Box(0.008, 0.058, 0.142, 0.975),
    "image": Box(0.142, 0.061, 0.574, 0.975),
    "description": Box(0.574, 0.058, 0.788, 0.975),
    "camera": Box(0.788, 0.058, 0.993, 0.975),
    "content_top": 0.058,
    "content_bottom": 0.975,
}

LAYOUT_MODES = [
    "auto",
    "seedance_wide_rows",
    "two_column_cards",
    "light_table_rows",
    "dark_table_rows",
]


@dataclass(frozen=True)
class SceneRegion:
    image: Box
    timing: Box
    description: Box
    camera: Box


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim == 4:
        image = image[0]
    array = image.detach().cpu().numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray(array[..., :3], "RGB")


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]


def _pad_to_batch(images: Iterable[Image.Image], background=(8, 13, 15)) -> torch.Tensor:
    prepared = [img.convert("RGB") for img in images]
    max_w = max(img.width for img in prepared)
    max_h = max(img.height for img in prepared)
    batch = []
    for img in prepared:
        canvas = Image.new("RGB", (max_w, max_h), background)
        canvas.paste(img, ((max_w - img.width) // 2, (max_h - img.height) // 2))
        batch.append(np.asarray(canvas).astype(np.float32) / 255.0)
    return torch.from_numpy(np.stack(batch, axis=0))


def _try_ocr(image: Image.Image, language: str) -> str:
    try:
        import pytesseract  # type: ignore
    except Exception:
        return ""

    try:
        gray = ImageOps.grayscale(image)
        gray = ImageOps.autocontrast(gray)
        scale = 2 if max(gray.size) < 1200 else 1
        if scale != 1:
            gray = gray.resize((gray.width * scale, gray.height * scale), Image.Resampling.LANCZOS)
        text = pytesseract.image_to_string(gray, lang=language or "deu+eng")
    except Exception:
        return ""
    return _clean_text(text)


def _clean_text(value: str) -> str:
    value = value.replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _strip_label(value: str, label: str) -> str:
    if not value:
        return ""
    return re.sub(rf"^\s*{re.escape(label)}\s*:?\s*", "", value, flags=re.IGNORECASE).strip()


def _parse_time_value(value: str) -> float:
    value = value.strip().lower().replace(",", ".")
    parts = value.split(":")
    if len(parts) == 2:
        return float(parts[0]) * 60.0 + float(parts[1])
    if len(parts) == 3:
        return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
    return float(value)


def _extract_timecode(text: str) -> Tuple[float, float, float, str]:
    cleaned = (text or "").replace("–", "-").replace("—", "-")
    patterns = [
        r"(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?)",
        r"(\d+(?:[.,]\d+)?)\s*-\s*(\d+(?:[.,]\d+)?)\s*(?:sec|sek|s)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            start = _parse_time_value(match.group(1))
            end = _parse_time_value(match.group(2))
        except Exception:
            continue
        if end > start:
            return start, end, end - start, match.group(0).strip()
    return 0.0, 0.0, 0.0, ""


def _format_seconds(value: float) -> str:
    if abs(value - round(value)) < 0.001:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _image_mean_luma(image: Image.Image) -> float:
    arr = np.asarray(ImageOps.grayscale(image.resize((64, 64), Image.Resampling.BILINEAR)))
    return float(arr.mean())


def _is_light_storyboard(image: Image.Image) -> bool:
    width, height = image.size
    sample = image.crop((0, int(height * 0.08), width, int(height * 0.92)))
    return _image_mean_luma(sample) > 135


def _choose_layout(image: Image.Image, layout_mode: str) -> str:
    if layout_mode != "auto":
        return layout_mode
    width, height = image.size
    aspect = width / max(1, height)
    if aspect > 1.0:
        return "seedance_wide_rows"
    if _is_light_storyboard(image):
        return "light_table_rows"
    return "two_column_cards" if aspect < 0.63 else "dark_table_rows"


def _auto_scene_count(layout: str, requested_rows: int, scene_count: int) -> int:
    if scene_count > 0:
        return scene_count
    if requested_rows > 0:
        return requested_rows
    return 6


def _regions_for_layout(layout: str, rows: int) -> List[SceneRegion]:
    regions: List[SceneRegion] = []
    if layout == "seedance_wide_rows":
        top = SEEDANCE_2_LAYOUT["content_top"]
        bottom = SEEDANCE_2_LAYOUT["content_bottom"]
        row_h = (bottom - top) / rows
        for index in range(rows):
            row_top = top + index * row_h
            row_bottom = top + (index + 1) * row_h
            regions.append(
                SceneRegion(
                    image=Box(SEEDANCE_2_LAYOUT["image"].left, row_top, SEEDANCE_2_LAYOUT["image"].right, row_bottom),
                    timing=Box(SEEDANCE_2_LAYOUT["left"].left, row_top, SEEDANCE_2_LAYOUT["left"].right, row_bottom),
                    description=Box(SEEDANCE_2_LAYOUT["description"].left, row_top, SEEDANCE_2_LAYOUT["description"].right, row_bottom),
                    camera=Box(SEEDANCE_2_LAYOUT["camera"].left, row_top, SEEDANCE_2_LAYOUT["camera"].right, row_bottom),
                )
            )
        return regions

    if layout == "two_column_cards":
        columns = 2
        grid_top = 0.052
        grid_bottom = 0.948
        grid_left = 0.01
        grid_right = 0.99
        gutter = 0.008
        card_rows = int(np.ceil(rows / columns))
        cell_w = (grid_right - grid_left - gutter) / columns
        cell_h = (grid_bottom - grid_top) / card_rows
        for index in range(rows):
            row = index // columns
            col = index % columns
            left = grid_left + col * (cell_w + gutter)
            right = left + cell_w
            top = grid_top + row * cell_h
            bottom = top + cell_h - 0.004
            image_bottom = top + (bottom - top) * 0.875
            regions.append(
                SceneRegion(
                    image=Box(left, top, right, image_bottom),
                    timing=Box(left, top, right, top + (bottom - top) * 0.12),
                    description=Box(left, top, right, image_bottom),
                    camera=Box(left, image_bottom, right, bottom),
                )
            )
        return regions

    if layout == "light_table_rows":
        top = 0.068
        bottom = 0.955
        row_h = (bottom - top) / rows
        for index in range(rows):
            row_top = top + index * row_h
            row_bottom = top + (index + 1) * row_h
            regions.append(
                SceneRegion(
                    image=Box(0.153, row_top, 0.512, row_bottom),
                    timing=Box(0.0, row_top, 0.153, row_bottom),
                    description=Box(0.512, row_top, 0.71, row_bottom),
                    camera=Box(0.71, row_top, 1.0, row_bottom),
                )
            )
        return regions

    if layout == "dark_table_rows":
        top = 0.064
        bottom = 0.945
        row_h = (bottom - top) / rows
        for index in range(rows):
            row_top = top + index * row_h
            row_bottom = top + (index + 1) * row_h
            regions.append(
                SceneRegion(
                    image=Box(0.16, row_top, 0.692, row_bottom),
                    timing=Box(0.008, row_top, 0.16, row_bottom),
                    description=Box(0.715, row_top, 0.995, row_bottom),
                    camera=Box(0.715, row_top + (row_bottom - row_top) * 0.64, 0.995, row_bottom),
                )
            )
        return regions

    return _regions_for_layout("seedance_wide_rows", rows)


class LTXStoryboardSplitter:
    """Split storyboard sheets into image and text regions across common GPT layouts."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "storyboard_image": ("IMAGE",),
                "rows": ("INT", {"default": 6, "min": 1, "max": 24, "step": 1}),
                "crop_inset_px": ("INT", {"default": 5, "min": 0, "max": 40, "step": 1}),
                "enable_ocr": ("BOOLEAN", {"default": True}),
                "ocr_language": ("STRING", {"default": "deu+eng", "multiline": False}),
                "project_title": ("STRING", {"default": "", "multiline": False}),
                "layout_mode": (LAYOUT_MODES, {"default": "auto"}),
                "scene_count": ("INT", {"default": 0, "min": 0, "max": 48, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "panel_images",
        "timing_setting_text_images",
        "description_text_images",
        "camera_text_images",
        "storyboard_json",
        "shot_prompts",
        "timing_csv",
    )
    FUNCTION = "split_storyboard"
    CATEGORY = "LTX/Storyboard"

    def split_storyboard(
        self,
        storyboard_image,
        rows: int = 6,
        crop_inset_px: int = 5,
        enable_ocr: bool = True,
        ocr_language: str = "deu+eng",
        project_title: str = "",
        layout_mode: str = "auto",
        scene_count: int = 0,
    ):
        pil = _tensor_to_pil(storyboard_image)
        width, height = pil.size
        resolved_layout = _choose_layout(pil, layout_mode)
        shot_count = _auto_scene_count(resolved_layout, rows, scene_count)
        regions = _regions_for_layout(resolved_layout, shot_count)

        panel_images: List[Image.Image] = []
        left_text_images: List[Image.Image] = []
        description_images: List[Image.Image] = []
        camera_images: List[Image.Image] = []
        shots: List[Dict[str, object]] = []

        for index, region in enumerate(regions):
            row_boxes = {
                "timing_setting": region.timing,
                "image": region.image,
                "description": region.description,
                "camera": region.camera,
            }

            crops = {
                name: pil.crop(box.pixels(width, height, crop_inset_px))
                for name, box in row_boxes.items()
            }
            panel_images.append(crops["image"])
            left_text_images.append(crops["timing_setting"])
            description_images.append(crops["description"])
            camera_images.append(crops["camera"])

            left_text = _try_ocr(crops["timing_setting"], ocr_language) if enable_ocr else ""
            description_text = _try_ocr(crops["description"], ocr_language) if enable_ocr else ""
            camera_text = _try_ocr(crops["camera"], ocr_language) if enable_ocr else ""

            shots.append(
                {
                    "shot": index + 1,
                    "timing_setting_text": left_text,
                    "description": _strip_label(description_text, "BESCHREIBUNG"),
                    "camera_movement": _strip_label(camera_text, "KAMERA / BEWEGUNG"),
                    "image_index": index,
                    "boxes": {
                        name: {
                            "left": box.left,
                            "top": box.top,
                            "right": box.right,
                            "bottom": box.bottom,
                        }
                        for name, box in row_boxes.items()
                    },
                }
            )

        payload = {
            "project_title": project_title,
            "source_size": {"width": width, "height": height},
            "layout": resolved_layout,
            "requested_layout": layout_mode,
            "scene_count": shot_count,
            "ocr_enabled": bool(enable_ocr),
            "ocr_language": ocr_language,
            "shots": shots,
        }

        shot_prompts = self._format_prompts(shots)
        timing_csv = self._format_timing_csv(shots)

        return (
            _pad_to_batch(panel_images),
            _pad_to_batch(left_text_images),
            _pad_to_batch(description_images),
            _pad_to_batch(camera_images),
            json.dumps(payload, ensure_ascii=False, indent=2),
            shot_prompts,
            timing_csv,
        )

    @staticmethod
    def _format_prompts(shots: List[Dict[str, object]]) -> str:
        lines = []
        for shot in shots:
            description = str(shot.get("description") or "").replace("\n", " ")
            camera = str(shot.get("camera_movement") or "").replace("\n", " ")
            if description or camera:
                lines.append(f"Shot {shot['shot']}: {description} Camera: {camera}".strip())
            else:
                lines.append(f"Shot {shot['shot']}:")
        return "\n".join(lines)

    @staticmethod
    def _format_timing_csv(shots: List[Dict[str, object]]) -> str:
        lines = ["shot,timing_setting_text,description,camera_movement"]
        for shot in shots:
            cells = [
                str(shot.get("shot", "")),
                str(shot.get("timing_setting_text", "")),
                str(shot.get("description", "")),
                str(shot.get("camera_movement", "")),
            ]
            escaped = ['"' + cell.replace('"', '""').replace("\n", " ").strip() + '"' for cell in cells]
            lines.append(",".join(escaped))
        return "\n".join(lines)


class LTXStoryboardAssetSaver:
    """Save split storyboard batches and text payloads into ComfyUI's output folder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panel_images": ("IMAGE",),
                "storyboard_json": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "shot_prompts": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "timing_csv": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "folder_name": ("STRING", {"default": "ltx_storyboard", "multiline": False}),
                "file_prefix": ("STRING", {"default": "shot", "multiline": False}),
            },
            "optional": {
                "timing_setting_text_images": ("IMAGE",),
                "description_text_images": ("IMAGE",),
                "camera_text_images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_folder",)
    FUNCTION = "save_assets"
    CATEGORY = "LTX/Storyboard"
    OUTPUT_NODE = True

    def save_assets(
        self,
        panel_images,
        storyboard_json: str,
        shot_prompts: str,
        timing_csv: str,
        folder_name: str = "ltx_storyboard",
        file_prefix: str = "shot",
        timing_setting_text_images=None,
        description_text_images=None,
        camera_text_images=None,
    ):
        output_root = self._output_root()
        safe_folder = self._safe_name(folder_name or "ltx_storyboard")
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = os.path.join(output_root, safe_folder, run_name)
        os.makedirs(target, exist_ok=True)

        safe_prefix = self._safe_name(file_prefix or "shot")
        self._save_image_batch(panel_images, target, safe_prefix)
        if timing_setting_text_images is not None:
            self._save_image_batch(timing_setting_text_images, target, f"{safe_prefix}_timing_text")
        if description_text_images is not None:
            self._save_image_batch(description_text_images, target, f"{safe_prefix}_description_text")
        if camera_text_images is not None:
            self._save_image_batch(camera_text_images, target, f"{safe_prefix}_camera_text")

        self._write_text(os.path.join(target, "storyboard.json"), storyboard_json)
        self._write_text(os.path.join(target, "shot_prompts.txt"), shot_prompts)
        self._write_text(os.path.join(target, "timing.csv"), timing_csv)
        return (target,)

    @staticmethod
    def _output_root() -> str:
        try:
            import folder_paths  # type: ignore

            return folder_paths.get_output_directory()
        except Exception:
            return os.path.join(os.getcwd(), "output")

    @staticmethod
    def _safe_name(value: str) -> str:
        value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
        return value.strip("._") or "storyboard"

    @staticmethod
    def _write_text(path: str, value: str) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value or "")

    @staticmethod
    def _save_image_batch(images: torch.Tensor, target: str, prefix: str) -> None:
        tensor = images.detach().cpu()
        if tensor.ndim == 3:
            tensor = tensor[None,]
        for index, image in enumerate(tensor, start=1):
            array = np.clip(image.numpy() * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(array[..., :3], "RGB").save(
                os.path.join(target, f"{prefix}_{index:02d}.png")
            )


class LTXStoryboardVideoPromptBuilder:
    """Build LTX 2.3 friendly prompts from the structured storyboard JSON."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "storyboard_json": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "global_style": (
                    "STRING",
                    {
                        "default": "cinematic, realistic motion, coherent lighting, clean temporal consistency",
                        "multiline": True,
                    },
                ),
                "continuity_notes": (
                    "STRING",
                    {
                        "default": "Preserve the source image identity, subject, outfit, environment, framing, aspect ratio and color palette. Animate only the intended action and camera movement.",
                        "multiline": True,
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "cartoon, video game, distorted hands, duplicated face, identity drift, bad anatomy, low quality, flicker, text, watermark",
                        "multiline": True,
                    },
                ),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60, "step": 1}),
                "seconds_per_scene": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 30.0, "step": 0.1}),
                "include_audio_cues": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = (
        "ltx_scene_prompts",
        "combined_prompt",
        "negative_prompt",
        "fallback_frames_per_scene",
        "total_frames",
        "scene_timing_json",
        "frame_counts_csv",
    )
    FUNCTION = "build_prompts"
    CATEGORY = "LTX/Storyboard"

    def build_prompts(
        self,
        storyboard_json: str,
        global_style: str,
        continuity_notes: str,
        negative_prompt: str,
        fps: int,
        seconds_per_scene: float,
        include_audio_cues: bool = True,
    ):
        shots = self._load_shots(storyboard_json)
        fallback_frames = max(1, int(round(float(fps) * float(seconds_per_scene))))
        scene_prompts = []
        timing_rows = []
        total_frames = 0
        for shot in shots:
            description = str(shot.get("description") or "").replace("\n", " ").strip()
            camera = str(shot.get("camera_movement") or "").replace("\n", " ").strip()
            timing = str(shot.get("timing_setting_text") or "").replace("\n", " ").strip()
            start_time, end_time, duration, timecode = _extract_timecode(timing)
            scene_seconds = duration if duration > 0 else float(seconds_per_scene)
            scene_frames = max(1, int(round(float(fps) * scene_seconds)))
            total_frames += scene_frames
            timing_rows.append(
                {
                    "shot": int(shot.get("shot", len(scene_prompts) + 1)),
                    "timecode": timecode,
                    "start_seconds": start_time if timecode else None,
                    "end_seconds": end_time if timecode else None,
                    "duration_seconds": scene_seconds,
                    "fps": int(fps),
                    "frames": scene_frames,
                    "used_fallback_duration": not bool(timecode),
                }
            )
            scene_prompts.append(
                self._format_ltx23_i2v_prompt(
                    shot_number=int(shot.get("shot", len(scene_prompts) + 1)),
                    description=description,
                    camera=camera,
                    timing=timing,
                    timecode=timecode,
                    start_time=start_time,
                    end_time=end_time,
                    global_style=global_style,
                    continuity_notes=continuity_notes,
                    seconds_per_scene=scene_seconds,
                    frames_per_scene=scene_frames,
                    fps=int(fps),
                    include_audio_cues=include_audio_cues,
                )
            )

        if not scene_prompts:
            total_frames = fallback_frames
            timing_rows = [
                {
                    "shot": 1,
                    "timecode": "",
                    "start_seconds": None,
                    "end_seconds": None,
                    "duration_seconds": float(seconds_per_scene),
                    "fps": int(fps),
                    "frames": fallback_frames,
                    "used_fallback_duration": True,
                }
            ]
            scene_prompts = [
                self._format_ltx23_i2v_prompt(
                    shot_number=1,
                    description="Animate the provided reference image with subtle natural motion.",
                    camera="Stable camera with gentle cinematic movement.",
                    timing="",
                    timecode="",
                    start_time=0.0,
                    end_time=0.0,
                    global_style=global_style,
                    continuity_notes=continuity_notes,
                    seconds_per_scene=float(seconds_per_scene),
                    frames_per_scene=fallback_frames,
                    fps=int(fps),
                    include_audio_cues=include_audio_cues,
                )
            ]

        combined_prompt = "\n\n".join(scene_prompts)
        frame_counts_csv = self._format_frame_counts_csv(timing_rows)
        return (
            "\n---SCENE---\n".join(scene_prompts),
            combined_prompt,
            negative_prompt.strip(),
            fallback_frames,
            total_frames,
            json.dumps(timing_rows, ensure_ascii=False, indent=2),
            frame_counts_csv,
        )

    @staticmethod
    def _format_ltx23_i2v_prompt(
        shot_number: int,
        description: str,
        camera: str,
        timing: str,
        timecode: str,
        start_time: float,
        end_time: float,
        global_style: str,
        continuity_notes: str,
        seconds_per_scene: float,
        frames_per_scene: int,
        fps: int,
        include_audio_cues: bool,
    ) -> str:
        action = description or "Animate the subject from the provided reference image with natural motion."
        camera_motion = camera or "Use a stable shot with subtle cinematic camera movement."
        framing = timing or "Use the framing visible in the source image."
        duration = f"{_format_seconds(seconds_per_scene)} seconds at {fps} fps, {frames_per_scene} frames"
        timeline = (
            f"Storyboard timecode: {timecode} ({_format_seconds(start_time)}s to {_format_seconds(end_time)}s)."
            if timecode
            else "Storyboard timecode: not detected, use the configured fallback clip length."
        )
        audio = (
            "Audio cues: match the visible action with natural ambience, subtle impacts, music hits or dialogue only when implied by the storyboard."
            if include_audio_cues
            else ""
        )
        parts = [
            f"LTX-2.3 Image-to-Video prompt for scene {shot_number}.",
            "Use the provided image as the first frame and visual reference.",
            continuity_notes.strip(),
            f"Action and motion: {action}",
            f"Camera movement: {camera_motion}",
            f"Shot timing and framing: {framing}",
            timeline,
            f"Clip length: {duration}.",
            f"Visual style: {global_style.strip()}",
            audio,
            "Avoid changing the main subject identity, composition, object count or scene layout unless the storyboard explicitly asks for it.",
        ]
        return " ".join(part for part in parts if part).strip()

    @staticmethod
    def _load_shots(storyboard_json: str) -> List[Dict[str, object]]:
        try:
            payload = json.loads(storyboard_json or "{}")
        except Exception:
            return []
        shots = payload.get("shots", [])
        if isinstance(shots, list):
            return [shot for shot in shots if isinstance(shot, dict)]
        return []

    @staticmethod
    def _format_frame_counts_csv(rows: List[Dict[str, object]]) -> str:
        lines = ["shot,timecode,start_seconds,end_seconds,duration_seconds,fps,frames,used_fallback_duration"]
        for row in rows:
            cells = [
                str(row.get("shot", "")),
                str(row.get("timecode", "")),
                "" if row.get("start_seconds") is None else str(row.get("start_seconds")),
                "" if row.get("end_seconds") is None else str(row.get("end_seconds")),
                str(row.get("duration_seconds", "")),
                str(row.get("fps", "")),
                str(row.get("frames", "")),
                str(row.get("used_fallback_duration", "")),
            ]
            escaped = ['"' + cell.replace('"', '""').replace("\n", " ").strip() + '"' for cell in cells]
            lines.append(",".join(escaped))
        return "\n".join(lines)


class LTXStoryboardSceneSelector:
    """Select one storyboard scene for an LTX 2.3 image-to-video render lane."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panel_images": ("IMAGE",),
                "ltx_scene_prompts": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "scene_timing_json": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "scene_index": ("INT", {"default": 1, "min": 1, "max": 48, "step": 1}),
                "fallback_frames": ("INT", {"default": 48, "min": 1, "max": 4096, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("scene_image", "positive_prompt", "negative_prompt", "frame_count", "scene_metadata")
    FUNCTION = "select_scene"
    CATEGORY = "LTX/Storyboard"

    def select_scene(
        self,
        panel_images,
        ltx_scene_prompts: str,
        negative_prompt: str,
        scene_timing_json: str,
        scene_index: int,
        fallback_frames: int,
    ):
        tensor = panel_images.detach().cpu()
        if tensor.ndim == 3:
            tensor = tensor[None,]
        safe_index = max(1, min(int(scene_index), int(tensor.shape[0]))) - 1
        scene_image = tensor[safe_index : safe_index + 1]

        prompts = [part.strip() for part in re.split(r"\n---SCENE---\n", ltx_scene_prompts or "") if part.strip()]
        positive = prompts[safe_index] if safe_index < len(prompts) else (prompts[0] if prompts else "")

        timing_rows = self._load_timing_rows(scene_timing_json)
        timing = timing_rows[safe_index] if safe_index < len(timing_rows) else {}
        frame_count = int(timing.get("frames") or fallback_frames)
        metadata = {
            "scene_index": safe_index + 1,
            "frame_count": frame_count,
            "timing": timing,
            "prompt": positive,
        }
        return (
            scene_image,
            positive,
            negative_prompt,
            frame_count,
            json.dumps(metadata, ensure_ascii=False, indent=2),
        )

    @staticmethod
    def _load_timing_rows(scene_timing_json: str) -> List[Dict[str, object]]:
        try:
            rows = json.loads(scene_timing_json or "[]")
        except Exception:
            return []
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
        return []


class LTXStoryboardVideoConcat:
    """Concatenate rendered scene videos with ffmpeg."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_video_folder": ("STRING", {"default": "", "multiline": False}),
                "file_pattern": ("STRING", {"default": "*.mp4", "multiline": False}),
                "output_filename": ("STRING", {"default": "storyboard_ltx23_final.mp4", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("final_video_path",)
    FUNCTION = "concat_videos"
    CATEGORY = "LTX/Storyboard"
    OUTPUT_NODE = True

    def concat_videos(self, scene_video_folder: str, file_pattern: str, output_filename: str):
        folder = os.path.abspath(scene_video_folder.strip() or os.getcwd())
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Scene video folder not found: {folder}")

        import glob

        files = sorted(glob.glob(os.path.join(folder, file_pattern.strip() or "*.mp4")))
        if not files:
            raise FileNotFoundError(f"No scene videos matched {file_pattern!r} in {folder}")

        safe_output = LTXStoryboardAssetSaver._safe_name(os.path.splitext(output_filename)[0]) + ".mp4"
        output_path = os.path.join(folder, safe_output)
        concat_list = os.path.join(folder, "_ltx23_concat_list.txt")
        with open(concat_list, "w", encoding="utf-8", newline="\n") as handle:
            for path in files:
                escaped = path.replace("\\", "/").replace("'", "'\\''")
                handle.write(f"file '{escaped}'\n")

        command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list,
            "-c",
            "copy",
            output_path,
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg was not found. Install ffmpeg or add it to PATH.") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"ffmpeg concat failed: {exc.stderr}") from exc

        return (output_path,)


class LTXStoryboardTextEncode:
    """Encode a linked STRING prompt without relying on CLIPTextEncode widget text."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "LTX/Storyboard"

    def encode(self, clip, text: str):
        tokens = clip.tokenize(text or "")
        return (clip.encode_from_tokens_scheduled(tokens),)


class LTXStoryboardPromptPreview:
    """Show the selected scene prompt in the ComfyUI output panel."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "preview"
    CATEGORY = "LTX/Storyboard"
    OUTPUT_NODE = True

    def preview(self, text: str):
        value = text or ""
        return {"ui": {"text": [value]}, "result": (value,)}


NODE_CLASS_MAPPINGS = {
    "LTXStoryboardSplitter": LTXStoryboardSplitter,
    "LTXStoryboardAssetSaver": LTXStoryboardAssetSaver,
    "LTXStoryboardVideoPromptBuilder": LTXStoryboardVideoPromptBuilder,
    "LTXStoryboardSceneSelector": LTXStoryboardSceneSelector,
    "LTXStoryboardVideoConcat": LTXStoryboardVideoConcat,
    "LTXStoryboardTextEncode": LTXStoryboardTextEncode,
    "LTXStoryboardPromptPreview": LTXStoryboardPromptPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXStoryboardSplitter": "LTX Storyboard Splitter",
    "LTXStoryboardAssetSaver": "LTX Storyboard Asset Saver",
    "LTXStoryboardVideoPromptBuilder": "LTX Storyboard Video Prompt Builder",
    "LTXStoryboardSceneSelector": "LTX Storyboard Scene Selector",
    "LTXStoryboardVideoConcat": "LTX Storyboard Video Concat",
    "LTXStoryboardTextEncode": "LTX Storyboard Text Encode",
    "LTXStoryboardPromptPreview": "LTX Storyboard Prompt Preview",
}
