# yuzu_anim_ground_trail_yuzu_full.py
# 32x56キャンバス + 文字上固定(HAPPY NEW YEARは出たら残る) + 残像なしGIF(背景塗り)
# ジャンプ後、転がっていく「地面」に "YUZU" を残す（置いたら最後まで残る）
#
# 使い方:
#   pip install pillow
#   python yuzu_anim_ground_trail_yuzu_full.py
#
# 出力:
#   yuzu.gif
#   yuzu_spritesheet.png

from PIL import Image, ImageDraw, ImageFont
import math

W, H = 56, 56
BG = (35, 35, 35, 255)

# --- Palette (RGBA) ---
OUTLINE = (60, 45, 20, 255)
Y_BASE  = (235, 186, 35, 255)
Y_LITE  = (255, 230, 120, 255)
Y_DARK  = (205, 140, 20, 255)
STEM    = (45, 140, 60, 255)

TEXT_WHITE = (255, 255, 255, 255)
TEXT_DARK  = (0, 0, 0, 255)  # 簡易縁取り用


def draw_yuzu(
    frame_img,
    cx,
    cy,
    r=11,
    rot=0.0,
    squash_y=1.0,
    squash_x=1.0,
    show_shadow=True,
    shadow_scale=1.0,
    shadow_alpha=80,
):
    draw = ImageDraw.Draw(frame_img)

    # Ground shadow
    if show_shadow:
        sh_w = max(1, int(r * 1.6 * squash_x * shadow_scale))
        sh_h = max(1, int(r * 0.6 * shadow_scale))
        sh_x0 = cx - sh_w // 2
        sh_y0 = cy + int(r * squash_y) - 2
        alpha = max(0, min(255, int(shadow_alpha)))
        draw.ellipse([sh_x0, sh_y0, sh_x0 + sh_w, sh_y0 + sh_h], fill=(0, 0, 0, alpha))

    # Body bbox
    rx = max(1, int(r * squash_x))
    ry = max(1, int(r * squash_y))
    x0, y0 = cx - rx, cy - ry
    x1, y1 = cx + rx, cy + ry

    # Fill base
    draw.ellipse([x0, y0, x1, y1], fill=Y_BASE)

    # Outline
    draw.ellipse([x0 - 1, y0 - 1, x1 + 1, y1 + 1], outline=OUTLINE)
    draw.ellipse([x0, y0, x1, y1], outline=OUTLINE)

    # Shading per-pixel
    px = frame_img.load()
    for y in range(y0 - 1, y1 + 2):
        for x in range(x0 - 1, x1 + 2):
            if 0 <= x < W and 0 <= y < H:
                nx = (x - cx) / rx
                ny = (y - cy) / ry
                if nx * nx + ny * ny <= 1.0:
                    light = (-nx - ny) * 0.5
                    shade = (nx + ny) * 0.5
                    if light > 0.35:
                        px[x, y] = Y_LITE
                    elif shade > 0.35:
                        px[x, y] = Y_DARK

    # Stem (rotating)
    stem_len = 4
    stem_cx = cx
    stem_cy = cy - ry + 3
    cosr, sinr = math.cos(rot), math.sin(rot)

    stem_points = []
    for t in range(-stem_len // 2, stem_len // 2 + 1):
        stem_points.append((0, t))
        stem_points.append((t, 0))
    stem_points.extend([(1, -2), (2, -3)])

    for (lx, ly) in stem_points:
        rxp = int(round(lx * cosr - ly * sinr))
        ryp = int(round(lx * sinr + ly * cosr))
        x = stem_cx + rxp
        y = stem_cy + ryp
        if 0 <= x < W and 0 <= y < H:
            px[x, y] = STEM

    # Rolling cue speck
    fx = int(round(cx + math.cos(rot + 0.8) * (r * 0.35)))
    fy = int(round(cy + math.sin(rot + 0.8) * (r * 0.20)))
    if 0 <= fx < W and 0 <= fy < H:
        px[fx, fy] = OUTLINE


def draw_text_with_outline(draw, x, y, text, font, fill=TEXT_WHITE):
    # 1px 縁取り
    for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        draw.text((x + ox, y + oy), text, fill=TEXT_DARK, font=font)
    draw.text((x, y), text, fill=fill, font=font)


def add_hny_text(frame_img):
    draw = ImageDraw.Draw(frame_img)
    font = ImageFont.load_default()

    line1 = "HAPPY"
    line2 = "NEW YEAR"

    b1 = draw.textbbox((0, 0), line1, font=font)
    b2 = draw.textbbox((0, 0), line2, font=font)
    w1, h1 = b1[2] - b1[0], b1[3] - b1[1]
    w2, _  = b2[2] - b2[0], b2[3] - b2[1]

    y1 = 2
    y2 = y1 + h1 + 1
    x1 = (W - w1) // 2
    x2 = (W - w2) // 2

    draw_text_with_outline(draw, x1, y1, line1, font)
    draw_text_with_outline(draw, x2, y2, line2, font)


def draw_ground_yuzu(frame_img, placed_letters, y_ground, baseline_offset=-7):
    """
    地面に残る 'YUZU' を描画
    placed_letters: [(char, x, y), ...]
    y は基本的に y_ground + baseline_offset（ユズの下あたり）想定
    """
    draw = ImageDraw.Draw(frame_img)
    font = ImageFont.load_default()

    for ch, x, y in placed_letters:
        # 画面外回避（雑でOK）
        x = max(0, min(W - 6, x))
        y = max(0, min(H - 10, y))
        draw_text_with_outline(draw, x, y, ch, font)


def make_frame(
    cx,
    cy,
    rot,
    squash_x=1.0,
    squash_y=1.0,
    show_shadow=True,
    shadow_scale=1.0,
    shadow_alpha=80,
    show_hny=False,
    ground_letters=None,
    y_ground=40,
):
    img = Image.new("RGBA", (W, H), BG)

    # 地面に残るYUZU（ユズより先に描くと“地面に描かれてる”感が出る）
    if ground_letters:
        draw_ground_yuzu(img, ground_letters, y_ground=y_ground)

    draw_yuzu(
        img,
        cx,
        cy,
        r=11,
        rot=rot,
        squash_x=squash_x,
        squash_y=squash_y,
        show_shadow=show_shadow,
        shadow_scale=shadow_scale,
        shadow_alpha=shadow_alpha,
    )

    if show_hny:
        add_hny_text(img)

    return img


def main():
    frames = []
    durations_ms = []

    roll_rots = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]

    # 文字領域を避けるための地面位置
    y_ground = 40

    # HNYは出たら残す
    hny_on = False

    # 地面に残る "YUZU"
    word = "YUZU"
    word_index = 0
    placed_letters = []  # [(char, x, y), ...]
    ground_y = y_ground + 2  # 地面付近（ユズに被りにくい位置）
    # ↑ もっと下にしたいなら +4〜+6

    # --- Entrance: roll in from left（文字なし） ---
    for i, x in enumerate([-10, -4, 2, 8, 14, 16]):
        frames.append(make_frame(x, y_ground, roll_rots[i % 4], show_hny=hny_on, ground_letters=placed_letters, y_ground=y_ground))
        durations_ms.append(90)

    # --- Jump prep (squash) ---
    frames.append(make_frame(16, y_ground, roll_rots[0], squash_x=1.15, squash_y=0.85, show_hny=hny_on, ground_letters=placed_letters, y_ground=y_ground))
    durations_ms.append(90)

    # --- Big jump ---
    jump_offsets = [-4, -10, -16, -16, -10, -4]
    jump_durs    = [70, 80, 110, 110, 80, 70]
    jump_rots    = [roll_rots[1], roll_rots[2], roll_rots[3], roll_rots[0], roll_rots[1], roll_rots[2]]

    for j, dy in enumerate(jump_offsets):
        h = abs(dy)
        frames.append(
            make_frame(
                16,
                y_ground + dy,
                jump_rots[j],
                show_shadow=True,
                shadow_scale=max(0.25, 1.0 - h / 20.0),
                shadow_alpha=int(max(18, 80 - h * 3)),
                show_hny=hny_on,
                ground_letters=placed_letters,
                y_ground=y_ground,
            )
        )
        durations_ms.append(jump_durs[j])

    # --- Land (squash) ---
    frames.append(make_frame(16, y_ground, roll_rots[2], squash_x=1.18, squash_y=0.82, show_hny=hny_on, ground_letters=placed_letters, y_ground=y_ground))
    durations_ms.append(110)

    # --- ここでHAPPY NEW YEARを出して、そのまま残す ---
    hny_on = True

    # 停止 + HNY（地面YUZUはまだ置かない）
    frames.append(make_frame(16, y_ground, roll_rots[2], show_hny=hny_on, ground_letters=placed_letters, y_ground=y_ground))
    durations_ms.append(220)

    # --- Drop（落下） ---
    frames.append(make_frame(16, y_ground + 10, roll_rots[3], show_shadow=False, show_hny=hny_on, ground_letters=placed_letters, y_ground=y_ground))
    durations_ms.append(90)

    # --- Exit: roll out to right（この“転がり中”に地面へYUZUを順に置く） ---
    path_out_x = [18, 22, 28, 36, 44]

    # どのタイミングで文字を置くか：退場の最初の4コマで 1文字ずつ置く
    # 置く位置は「今のユズの少し後ろ」にする（trailっぽく）
    for i, x in enumerate(path_out_x):
        # i=0..3 で Y U Z U を置く
        if word_index < len(word) and i < 4:
            # ユズの後ろ（左側）に置く。-10 は好みで調整してOK。
            place_x = x - 10
            placed_letters.append((word[word_index], place_x, ground_y))
            word_index += 1

        rot = roll_rots[(3 - (i % 4)) % 4]
        frames.append(make_frame(x, y_ground, rot, show_hny=hny_on, ground_letters=placed_letters, y_ground=y_ground))
        durations_ms.append(90)

    # --- Save GIF (stable) ---
    frames_p = [fr.convert("P", palette=Image.ADAPTIVE, colors=256) for fr in frames]
    frames_p[0].save(
        "yuzu.gif",
        save_all=True,
        append_images=frames_p[1:],
        duration=durations_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )

    # --- Save sprite sheet ---
    sheet = Image.new("RGBA", (W * len(frames), H), BG)
    for i, fr in enumerate(frames):
        sheet.paste(fr, (i * W, 0))
    sheet.save("yuzu_spritesheet.png")

    print(f"Saved: yuzu.gif / yuzu_spritesheet.png (frames={len(frames)})")


if __name__ == "__main__":
    main()
