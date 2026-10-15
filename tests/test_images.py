"""Preprocessing and fingerprints: the properties the near-duplicate stage
depends on."""
import io

from PIL import Image

from matcher import fingerprint as fp
from matcher import images as im


def noisy(size=(256, 256)):
    return Image.effect_noise(size, 40).convert("RGB")


def test_hashes_survive_downscale_and_jpeg():
    base = noisy()
    buf = io.BytesIO()
    base.resize((128, 128)).save(buf, "JPEG", quality=55)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")
    distance = fp.dup_distance(
        fp.dhash(base), fp.phash(base), fp.dhash(recompressed), fp.phash(recompressed)
    )
    assert distance <= 12


def test_unrelated_images_are_far_apart():
    a, b = noisy(), Image.new("RGB", (256, 256), (200, 30, 40))
    assert fp.dup_distance(fp.dhash(a), fp.phash(a), fp.dhash(b), fp.phash(b)) > 12


def test_dup_score_is_monotonic_and_bounded():
    assert fp.dup_score(0) == 1.0
    assert fp.dup_score(64) == 0.0
    assert fp.dup_score(4) > fp.dup_score(10)


def test_uniform_borders_are_stripped_back_to_the_content():
    content = noisy((200, 200))
    padded = Image.new("RGB", (400, 500), (255, 255, 255))
    padded.paste(content, (100, 150))
    assert im.strip_uniform_borders(padded).size == (200, 200)


def test_border_stripping_leaves_a_full_bleed_image_alone():
    base = noisy((300, 300))
    assert im.strip_uniform_borders(base).size == base.size


def test_crop_ensemble_covers_chrome_and_body():
    names = {c.name for c in im.crop_ensemble(noisy((400, 600)))}
    assert {"full", "chrome_stripped", "upper", "lower"} <= names


def _png(image):
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


def test_load_accepts_raw_bytes_and_returns_rgb():
    loaded = im.load(_png(noisy((64, 96))))
    assert loaded.size == (64, 96)
    assert loaded.mode == "RGB"


def test_load_bounds_very_large_images():
    loaded = im.load(_png(noisy((2400, 1800))))
    assert max(loaded.size) <= im.MAX_EDGE


def test_colour_families():
    cases = {
        (190, 24, 40): "red", (10, 10, 12): "black", (250, 250, 250): "white",
        (20, 30, 90): "navy", (40, 120, 60): "green", (38, 84, 176): "blue",
    }
    for rgb, expected in cases.items():
        assert fp.colour_family(Image.new("RGB", (80, 80), rgb)) == expected
