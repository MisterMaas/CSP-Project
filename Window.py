from collections import deque  # <── Added for frame buffering
import imageio  # <── Added for video/gif saving
import numpy as np
import numpy.random as random
import pygame
from Model import Model
import math

# ── colours ────────────────────────────────────────────────────────────────
BG = (15, 15, 20)
PANEL_BG = (22, 22, 30)
ACCENT = (0, 210, 180)
BTN_GREEN = (30, 160, 90)
BTN_ORANGE = (210, 110, 30)
BTN_BLUE = (40, 120, 200)
GRAPH_BG = (18, 18, 25)
TEXT = (210, 220, 230)
TEXT_DIM = (110, 120, 135)
DIVIDER = (40, 42, 55)

# Cap history length to prevent unbounded list growth
MAX_HIST = 500


# ── heatmap colour helper ────────────────────────────────────────────────────
def resource_to_color(value: float, max_val: float) -> tuple:
    """Map a resource value seamlessly from black (0) to bright green (max)."""
    if max_val <= 0 or value <= 0:
        return (0, 0, 0)
    t = min(1.0, value / max_val)
    return (0, int(t * 255), 0)


# ── tiny UI widgets ─────────────────────────────────────────────────────────
class Slider:
    H = 6

    def __init__(self, label, x, y, w, lo, hi, init, fmt=str, callback=None):
        self.label = label
        self.track = pygame.Rect(x, y + 18, w, self.H)
        self.lo, self.hi = lo, hi
        self.value = init
        self.fmt = fmt
        self.callback = callback
        self.dragging = False

    @property
    def _handle_x(self):
        t = (self.value - self.lo) / (self.hi - self.lo)
        return int(self.track.x + t * self.track.w)

    def draw(self, surf, font):
        hx = self._handle_x
        pygame.draw.rect(surf, (50, 52, 65), self.track, border_radius=3)
        filled = pygame.Rect(self.track.x, self.track.y, hx - self.track.x, self.H)
        pygame.draw.rect(surf, ACCENT, filled, border_radius=3)
        pygame.draw.circle(surf, (230, 235, 245), (hx, self.track.centery), 8)
        pygame.draw.circle(surf, ACCENT, (hx, self.track.centery), 5)
        surf.blit(
            font.render(f"{self.label}:  {self.fmt(self.value)}", True, TEXT),
            (self.track.x, self.track.y - 18),
        )

    def handle(self, event, offset):
        """offset = (panel_screen_x, panel_screen_y)"""
        hx = self._handle_x + offset[0]
        hy = self.track.centery + offset[1]
        if event.type == pygame.MOUSEBUTTONDOWN:
            if abs(event.pos[0] - hx) < 12 and abs(event.pos[1] - hy) < 12:
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            t = (event.pos[0] - offset[0] - self.track.x) / self.track.w
            self.value = self.lo + t * (self.hi - self.lo)
            self.value = max(self.lo, min(self.hi, self.value))
            if isinstance(self.lo, int) and isinstance(self.hi, int):
                self.value = int(round(self.value))
            if self.callback:
                self.callback(self.value)


class Button:

    def __init__(self, label, x, y, w, h, callback, color=BTN_GREEN):
        self.label = label
        self.rect = pygame.Rect(x, y, w, h)
        self.color = color
        self.callback = callback
        self._hover = False

    def draw(self, surf, font):
        col = (
            tuple(min(255, c + 25) for c in self.color)
            if self._hover
            else self.color
        )
        pygame.draw.rect(surf, col, self.rect, border_radius=5)
        t = font.render(self.label, True, (255, 255, 255))
        surf.blit(t, t.get_rect(center=self.rect.center))

    def handle(self, event, offset):
        sr = self.rect.move(offset)
        if event.type == pygame.MOUSEMOTION:
            self._hover = sr.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and sr.collidepoint(event.pos):
            self.callback()


# ── main window ─────────────────────────────────────────────────────────────
class Window:
    PANEL_W = 240
    GRAPH_H = 180
    FPS = 60

    def __init__(self, win_w=1200, win_h=820):
        pygame.init()
        self.screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
        pygame.display.set_caption("Evolution of Evolvable GRN")
        self.clock = pygame.time.Clock()

        self.W, self.H = win_w, win_h

        self.model = None
        self.auto_run = False
        self.sim_speed = 10
        self._auto_acc = 0.0

        # Graph 1 Data History
        self.min_hist = []
        self.mean_hist = []
        self.std_hist = []  # <── Added for Hamming Distance STD

        # Graph 2 Data History
        self.org_mean_hist = []
        self.org_max_hist = []
        self.org_std_hist = []

        self.font = pygame.font.SysFont("consolas", 13)
        self.font_b = pygame.font.SysFont("consolas", 13, bold=True)

        self.id_colors = {}
        self.frame_buffer = deque(maxlen=100)

        self._build_ui()
        self._reset()
        self._update_dimensions(win_w, win_h)

    def _update_dimensions(self, w, h):
        self.W, self.H = w, h
        self.grid_w = w - self.PANEL_W
        self.grid_h = h - self.GRAPH_H

        self.sub_graph_w = self.grid_w // 2

        self._grid_surf = pygame.Surface((self.grid_w, self.grid_h))
        self._graph1_surf = pygame.Surface((self.sub_graph_w, self.GRAPH_H))
        self._graph2_surf = pygame.Surface((self.sub_graph_w, self.GRAPH_H))
        self._panel_surf = pygame.Surface((self.PANEL_W, self.H))

        self._build_ui()

    def _build_ui(self):
        px, pw = 18, self.PANEL_W - 36

        self.sliders = [
            Slider("Fitness Power", px, 25, pw, 1.0, 25.0, 10.0, fmt=lambda v: f"{v:.1f}",
                   callback=lambda v: self.model and setattr(self.model, "FitnessPower", v)),
            Slider("Mut. Factor", px, 80, pw, 1.0, 150.0, 75.0, fmt=lambda v: f"{v:.1f}",
                   callback=lambda v: self.model and setattr(self.model, "MutationFactor", v)),
            Slider("Mean Resource", px, 135, pw, 0.0, 5.0, 2.0, fmt=lambda v: f"{v:.1f}"),
            Slider("SD Resource", px, 190, pw, 0.0, 5.0, 2.0, fmt=lambda v: f"{v:.1f}"),
            Slider("Regen Rate", px, 245, pw, 0.0, 0.1, 0.05, fmt=lambda v: f"{v:.3f}",
                   callback=lambda v: self.model and setattr(self.model, "RegenRate", v)),
            Slider("Sim Speed", px, 300, pw, 1, 500, 10, callback=lambda v: setattr(self, "sim_speed", v)),
            Slider("Div. Threshold", px, 355, pw, 5, 50, 15,
                   callback=lambda v: self.model and setattr(self.model, "DivisionThreshold", v)),
            Slider("Div. Timesteps", px, 410, pw, 1, 30, 5,
                   callback=lambda v: self.model and setattr(self.model, "DivisionTimeSteps", v)),
        ]

        bh, gap = 34, 6
        by = 470
        self.buttons = [
            Button("Reset", px, by, pw, bh, self._reset, BTN_GREEN),
            Button("Step", px, by + (bh + gap), pw, bh, self._step, BTN_GREEN),
            Button("▶  Auto Run", px, by + 2 * (bh + gap), pw, bh, self._toggle_auto, BTN_GREEN),
            Button("Switch Target", px, by + 3 * (bh + gap), pw, bh, self._switch, BTN_BLUE),
            Button("🎬 Save Video", px, by + 4 * (bh + gap), pw, bh, self._save_video, BTN_BLUE),
        ]
        self._auto_btn = self.buttons[2]

    def _reset(self):
        fp = self.sliders[0].value if hasattr(self, "sliders") else 10
        mf = self.sliders[1].value if hasattr(self, "sliders") else 1.0
        mr = self.sliders[2].value if hasattr(self, "sliders") else 5.0
        sdr = self.sliders[3].value if hasattr(self, "sliders") else 3.0
        rr = self.sliders[4].value if hasattr(self, "sliders") else 0.05
        d_th = self.sliders[6].value if hasattr(self, "sliders") else 15
        d_ts = self.sliders[7].value if hasattr(self, "sliders") else 5

        self.model = Model(
            fitness_power=fp,
            mutation_factor=mf,
            mean_resource=mr,
            sd_recourse=sdr,
            regen_rate=rr,
            division_thres=d_th,
            division_timesteps=d_ts,
        )
        self.min_hist.clear()
        self.mean_hist.clear()
        self.std_hist.clear()  # <── Clear new history array
        self.org_mean_hist.clear()
        self.org_max_hist.clear()
        self.org_std_hist.clear()
        self._auto_acc = 0.0
        self.id_colors.clear()
        self.frame_buffer.clear()

    def _step(self):
        if self.model is None:
            return
        self.model.ExecuteStep()

        # History Graph 1
        self.min_hist.append(self.model.MinimalDistance)
        self.mean_hist.append(self.model.MeanDistance)
        self.std_hist.append(getattr(self.model, "STDDistance", 0.0))  # <── Fetches genetic distance STD

        # History Graph 2
        self.org_mean_hist.append(self.model.MeanOrgSize)
        self.org_max_hist.append(self.model.MaxOrgSize)
        self.org_std_hist.append(self.model.STDOrgSize)

        if len(self.min_hist) > MAX_HIST:
            self.min_hist = self.min_hist[-MAX_HIST:]
            self.mean_hist = self.mean_hist[-MAX_HIST:]
            self.std_hist = self.std_hist[-MAX_HIST:]
            self.org_mean_hist = self.org_mean_hist[-MAX_HIST:]
            self.org_max_hist = self.org_max_hist[-MAX_HIST:]
            self.org_std_hist = self.org_std_hist[-MAX_HIST:]

    def _toggle_auto(self):
        self.auto_run = not self.auto_run
        if self.auto_run:
            self._auto_btn.label = "⏸  Pause"
            self._auto_btn.color = BTN_ORANGE
        else:
            self._auto_btn.label = "▶  Auto Run"
            self._auto_btn.color = BTN_GREEN

    def _switch(self):
        if self.model:
            self.model.SwitchTarget()

    def _save_video(self):
        if not self.frame_buffer:
            print("No frames captured yet!")
            return
        print(f"Exporting {len(self.frame_buffer)} frames...")
        output_filename = "simulation_history.gif"
        try:
            imageio.mimsave(output_filename, list(self.frame_buffer), fps=20)
            print(f"Successfully saved GIF to {output_filename}")
        except Exception as e:
            print(f"Failed to save GIF: {e}")

    def _get_color_for_id(self, cell_id):
        if cell_id not in self.id_colors:
            rng = np.random.default_rng(int(cell_id))
            if rng.random() < 0.5:
                r = rng.integers(130, 256)
                g = 0
                b = rng.integers(0, 141)
            else:
                r = rng.integers(0, 121)
                g = 0
                b = rng.integers(130, 256)
            self.id_colors[cell_id] = (int(r), int(g), int(b))
        return self.id_colors[cell_id]

    def _render_grid(self):
        surf = self._grid_surf
        surf.fill(BG)

        if self.model is None or self.model.Grid is None:
            msg = self.font.render("No model loaded", True, TEXT_DIM)
            surf.blit(msg, msg.get_rect(center=(self.grid_w // 2, self.grid_h // 2)))
            return surf

        xs, ys = self.model.xSize, self.model.ySize
        cell_w = self.grid_w // xs
        cell_h = self.grid_h // ys
        cell_size = max(2, min(cell_w, cell_h))

        offset_x = (self.grid_w - xs * cell_size) // 2
        offset_y = (self.grid_h - ys * cell_size) // 2

        resource_grid = self.model.Grid
        max_val = float(np.max(resource_grid)) if resource_grid.size > 0 else 1.0

        for i in range(xs):
            for j in range(ys):
                color = resource_to_color(resource_grid[i, j], max_val)
                rect = pygame.Rect(offset_x + i * cell_size, offset_y + j * cell_size, cell_size, cell_size)
                pygame.draw.rect(surf, color, rect)

        max_genes = getattr(self.model, "NumberOfGenes", 20)
        border_thickness = max(1, min(4, cell_size // 4))

        for cell in self.model.Cells:
            i, j = cell.iPos, cell.jPos
            rect = pygame.Rect(offset_x + i * cell_size, offset_y + j * cell_size, cell_size, cell_size)

            dist = getattr(cell, "HammingDistance", 0)
            dist_ratio = np.clip(float(dist) / max_genes, 0.0, 1.0)

            clear_orange = (255, 120, 10)
            gray_orange = (110, 90, 75)

            r = int(clear_orange[0] + dist_ratio * (gray_orange[0] - clear_orange[0]))
            g = int(clear_orange[1] + dist_ratio * (gray_orange[1] - clear_orange[1]))
            b = int(clear_orange[2] + dist_ratio * (gray_orange[2] - clear_orange[2]))
            border_color = (r, g, b)

            if cell.IsStable:
                body_color = self._get_color_for_id(cell.ID)
                pygame.draw.rect(surf, body_color, rect)
                pygame.draw.rect(surf, border_color, rect, border_thickness)
            else:
                r, g, b = self._get_color_for_id(cell.ID)
                dim_color = (r // 3, g // 3, b // 3)
                pygame.draw.rect(surf, dim_color, rect)
                pygame.draw.rect(surf, (140, 40, 40), rect, border_thickness)

        self._draw_heatmap_legend(surf, offset_x, offset_y + ys * cell_size, max_val)
        return surf

    def _draw_heatmap_legend(self, surf, x, y, max_val):
        bar_w, bar_h = 120, 8
        lx, ly = x + 4, y - bar_h - 18
        if ly < 0: return
        for px in range(bar_w):
            color = resource_to_color(px / bar_w * max_val, max_val)
            pygame.draw.line(surf, color, (lx + px, ly), (lx + px, ly + bar_h))
        pygame.draw.rect(surf, DIVIDER, (lx, ly, bar_w, bar_h), 1)
        surf.blit(self.font.render("0", True, TEXT_DIM), (lx, ly + bar_h + 2))
        label = self.font.render(f"{max_val:.0f}", True, TEXT_DIM)
        surf.blit(label, (lx + bar_w - label.get_width(), ly + bar_h + 2))
        mid = self.font.render("Resource", True, TEXT_DIM)
        surf.blit(mid, (lx + bar_w // 2 - mid.get_width() // 2, ly - 13))

    # ── Error Band Rendering Helper ──────────────────────────────────────────
    def _draw_error_band(self, surf, pad, gw, gh, max_val, mean_hist, std_hist, color_rgb, alpha=45):
        """Renders a semi-transparent standard deviation band around a mean sequence."""
        n = len(mean_hist)
        if n < 2: return

        # Form a list of points representing the upper edge and lower edge
        upper_pts = []
        lower_pts = []

        for i in range(n):
            x = pad + int(i / (n - 1) * gw)

            # Map values with clipping bounds to prevent drawing out of bounds
            y_high = pad + gh - int(max(0, min(max_val, mean_hist[i] + std_hist[i])) / max_val * gh)
            y_low = pad + gh - int(max(0, min(max_val, mean_hist[i] - std_hist[i])) / max_val * gh)

            upper_pts.append((x, y_high))
            lower_pts.append((x, y_low))

        # Join paths together to create a solid polygon wrap
        polygon_points = upper_pts + lower_pts[::-1]

        # Use an alpha surface layer to get transparent blending in Pygame
        temp_surface = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        pygame.draw.polygon(temp_surface, (*color_rgb, alpha), polygon_points)
        surf.blit(temp_surface, (0, 0))

    # ── Graph 1: Distance Metrics (with Shaded STD Band) ────────────────────
    def _render_graph1(self):
        surf = self._graph1_surf
        surf.fill(GRAPH_BG)

        pad = 36
        gw = self.sub_graph_w - 2 * pad
        gh = self.GRAPH_H - 2 * pad
        max_val = 20

        pygame.draw.line(surf, DIVIDER, (pad, pad), (pad, pad + gh))
        pygame.draw.line(surf, DIVIDER, (pad, pad + gh), (pad + gw, pad + gh))

        for v in range(0, 21, 5):
            y = pad + gh - int(v / max_val * gh)
            pygame.draw.line(surf, DIVIDER, (pad - 4, y), (pad + gw, y))
            surf.blit(self.font.render(str(v), True, TEXT_DIM), (2, y - 7))

        n = len(self.min_hist)
        if n < 2:
            surf.blit(self.font.render("Waiting for data…", True, TEXT_DIM), (pad + 8, pad + gh // 2))
            return surf

        # 1. Draw Shaded STD Deviation Band (Drawn first so it stays behind lines)
        if len(self.std_hist) == n:
            self._draw_error_band(surf, pad, gw, gh, max_val, self.mean_hist, self.std_hist, (255, 160, 40))

        def px(i, v):
            x = pad + int(i / (n - 1) * gw)
            y = pad + gh - int(v / max_val * gh)
            return x, y

        # 2. Draw lines
        for i in range(n - 1):
            pygame.draw.line(surf, (0, 210, 220), px(i, self.min_hist[i]), px(i + 1, self.min_hist[i + 1]), 2)
            pygame.draw.line(surf, (255, 160, 40), px(i, self.mean_hist[i]), px(i + 1, self.mean_hist[i + 1]), 2)

        # Legend Metrics
        pygame.draw.line(surf, (0, 210, 220), (pad, 8), (pad + 20, 8), 2)
        surf.blit(self.font.render("min dist", True, (0, 210, 220)), (pad + 24, 1))

        # Combined line and band box icon marker for legend
        pygame.draw.rect(surf, (255, 160, 40, 60), (pad + 90, 4, 20, 8))
        pygame.draw.line(surf, (255, 160, 40), (pad + 90, 8), (pad + 110, 8), 2)
        surf.blit(self.font.render("mean dist (±std)", True, (255, 160, 40)), (pad + 114, 1))
        return surf

    # ── Graph 2: Organism Sizes (with Shaded STD Band) ──────────────────────
    def _render_graph2(self):
        surf = self._graph2_surf
        surf.fill(GRAPH_BG)

        pad = 36
        gw = self.sub_graph_w - 2 * pad
        gh = self.GRAPH_H - 2 * pad

        n = len(self.org_mean_hist)
        if n < 2:
            pygame.draw.line(surf, DIVIDER, (pad, pad), (pad, pad + gh))
            pygame.draw.line(surf, DIVIDER, (pad, pad + gh), (pad + gw, pad + gh))
            surf.blit(self.font.render("Waiting for data…", True, TEXT_DIM), (pad + 8, pad + gh // 2))
            return surf

        highest_data_point = max(max(self.org_max_hist), 5)
        max_val = int(math.ceil(highest_data_point / 5.0) * 5)

        pygame.draw.line(surf, DIVIDER, (pad, pad), (pad, pad + gh))
        pygame.draw.line(surf, DIVIDER, (pad, pad + gh), (pad + gw, pad + gh))

        step_size = max(1, max_val // 4)
        for v in range(0, max_val + 1, step_size):
            y = pad + gh - int(v / max_val * gh)
            pygame.draw.line(surf, DIVIDER, (pad - 4, y), (pad + gw, y))
            surf.blit(self.font.render(str(v), True, TEXT_DIM), (2, y - 7))

        # 1. Draw Shaded STD Deviation Band around the Organism Mean Size
        self._draw_error_band(surf, pad, gw, gh, max_val, self.org_mean_hist, self.org_std_hist, (50, 220, 100))

        def px(i, v):
            x = pad + int(i / (n - 1) * gw)
            y = pad + gh - int(v / max_val * gh)
            return x, y

        # 2. Draw metric lines
        for i in range(n - 1):
            pygame.draw.line(surf, (240, 50, 100), px(i, self.org_max_hist[i]), px(i + 1, self.org_max_hist[i + 1]), 2)
            pygame.draw.line(surf, (50, 220, 100), px(i, self.org_mean_hist[i]), px(i + 1, self.org_mean_hist[i + 1]),
                             2)

        # Draw Legend Labels
        pygame.draw.line(surf, (240, 50, 100), (pad, 8), (pad + 20, 8), 2)
        surf.blit(self.font.render("max size", True, (240, 50, 100)), (pad + 24, 1))

        pygame.draw.rect(surf, (50, 220, 100, 60), (pad + 100, 4, 20, 8))
        pygame.draw.line(surf, (50, 220, 100), (pad + 100, 8), (pad + 120, 8), 2)
        surf.blit(self.font.render("mean size (±std)", True, (50, 220, 100)), (pad + 124, 1))

        return surf

    # ── side panel ───────────────────────────────────────────────────────────
    def _render_panel(self):
        surf = self._panel_surf
        surf.fill(PANEL_BG)

        hdr = self.font_b.render("GRN  EVOLUTION", True, ACCENT)
        surf.blit(hdr, hdr.get_rect(centerx=self.PANEL_W // 2, y=8))
        pygame.draw.line(surf, ACCENT, (12, 26), (self.PANEL_W - 12, 26))

        for s in self.sliders: s.draw(surf, self.font)
        for b in self.buttons: b.draw(surf, self.font)

        sy = 650
        pygame.draw.line(surf, DIVIDER, (12, sy - 8), (self.PANEL_W - 12, sy - 8))

        if self.model is None:
            surf.blit(self.font.render("No model loaded", True, TEXT_DIM), (18, sy))
        else:
            stats = [
                ("Timestep", getattr(self.model, "Timestep", 0)),
                ("Population", getattr(self.model, "TotalPopulation", "—")),
                ("Min dist", getattr(self.model, "MinimalDistance", "—")),
                ("Mean dist", f"{getattr(self.model, 'MeanDistance', 0):.2f}"),
                ("Target", ("A" if np.array_equal(getattr(self.model, "Target", None),
                                                  getattr(self.model, "TargetA", None)) else "B")),
            ]
            for label, val in stats:
                surf.blit(self.font.render(f"{label:<12} {val}", True, TEXT), (18, sy))
                sy += 20
        return surf

    # ── main loop ────────────────────────────────────────────────────────────
    def run(self):
        while True:
            panel_origin = (self.grid_w, 0)
            dt = self.clock.tick(self.FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.VIDEORESIZE:
                    self._update_dimensions(event.w, event.h)

                for s in self.sliders: s.handle(event, panel_origin)
                for b in self.buttons: b.handle(event, panel_origin)

            if self.auto_run and self.model:
                self._auto_acc += dt
                interval = 1000 / max(1, self.sim_speed)
                self._auto_acc = min(self._auto_acc, interval * 2)
                while self._auto_acc >= interval:
                    self._step()
                    self._auto_acc -= interval

            self.screen.fill(BG)
            self.screen.blit(self._render_grid(), (0, 0))

            self.screen.blit(self._render_graph1(), (0, self.grid_h))
            self.screen.blit(self._render_graph2(), (self.sub_graph_w, self.grid_h))

            self.screen.blit(self._render_panel(), panel_origin)

            pygame.draw.line(self.screen, DIVIDER, (0, self.grid_h), (self.grid_w, self.grid_h))
            pygame.draw.line(self.screen, DIVIDER, (self.sub_graph_w, self.grid_h), (self.sub_graph_w, self.H))
            pygame.draw.line(self.screen, DIVIDER, (self.grid_w, 0), (self.grid_w, self.H))

            frame_str = pygame.image.tostring(self.screen, "RGB")
            frame_np = np.frombuffer(frame_str, dtype=np.uint8).reshape(self.H, self.W, 3)
            self.frame_buffer.append(frame_np)

            pygame.display.flip()


if __name__ == '__main__':
    Window().run()