import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

FIELD_LENGTH = 120.0
FIELD_WIDTH = 80.0

DT = 0.05
V = 6.0
W = 2.0

NUM_PARTICLES = 100
RANDOM_PARTICLE_RATIO = 0.02
MOTION_NOISE = np.diag([0.5, 0.5, 0.05])
LIDAR_NOISE = 1.0

NUM_BEAMS = 8
FOV = np.deg2rad(180)
MAX_RANGE = 25.0

GRID_RES = 0.5
LOG_ODD_OCC = 0.85
LOG_ODD_FREE = -0.4
LOG_ODD_MIN = -5.0
LOG_ODD_MAX = 5.0


class SoccerField:
    def __init__(self):
        L = FIELD_LENGTH
        W = FIELD_WIDTH

        self.lines = []

        self.lines += [
            ((-L / 2, -W / 2), (L / 2, -W / 2)),
            ((L / 2, -W / 2), (L / 2, W / 2)),
            ((L / 2, W / 2), (-L / 2, W / 2)),
            ((-L / 2, W / 2), (-L / 2, -W / 2)),
        ]

        self.lines.append(((0, -W / 2), (0, W / 2)))

        PA_D = 16.5
        PA_W = 40.3

        self.lines += [
            ((-L / 2, -PA_W / 2), (-L / 2 + PA_D, -PA_W / 2)),
            ((-L / 2 + PA_D, -PA_W / 2), (-L / 2 + PA_D, PA_W / 2)),
            ((-L / 2 + PA_D, PA_W / 2), (-L / 2, PA_W / 2)),
        ]

        self.lines += [
            ((L / 2 - PA_D, -PA_W / 2), (L / 2, -PA_W / 2)),
            ((L / 2 - PA_D, -PA_W / 2), (L / 2 - PA_D, PA_W / 2)),
            ((L / 2 - PA_D, PA_W / 2), (L / 2, PA_W / 2)),
        ]

        GA_D = 5.5
        GA_W = 18.3

        self.lines += [
            ((-L / 2, -GA_W / 2), (-L / 2 + GA_D, -GA_W / 2)),
            ((-L / 2 + GA_D, -GA_W / 2), (-L / 2 + GA_D, GA_W / 2)),
            ((-L / 2 + GA_D, GA_W / 2), (-L / 2, GA_W / 2)),
        ]

        self.lines += [
            ((L / 2 - GA_D, -GA_W / 2), (L / 2, -GA_W / 2)),
            ((L / 2 - GA_D, -GA_W / 2), (L / 2 - GA_D, GA_W / 2)),
            ((L / 2 - GA_D, GA_W / 2), (L / 2, GA_W / 2)),
        ]

        R = 9.15
        N = 40
        for i in range(N):
            a1 = 2 * np.pi * i / N
            a2 = 2 * np.pi * (i + 1) / N
            p1 = (R * np.cos(a1), R * np.sin(a1))
            p2 = (R * np.cos(a2), R * np.sin(a2))
            self.lines.append((p1, p2))

    def draw(self, ax):
        for p1, p2 in self.lines:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="black")
        ax.set_aspect("equal")
        ax.set_xlim(-FIELD_LENGTH / 2 - 5, FIELD_LENGTH / 2 + 5)
        ax.set_ylim(-FIELD_WIDTH / 2 - 5, FIELD_WIDTH / 2 + 5)
        ax.set_title("Soccer Field (Known Map)")


class Robot:
    def __init__(self):
        self.x = FIELD_LENGTH / 2 - 6
        self.y = -FIELD_WIDTH / 2 + 6
        self.theta = np.pi
        self.v = 0.0
        self.w = 0.0

    def step(self):
        self.x += self.v * np.cos(self.theta) * DT
        self.y += self.v * np.sin(self.theta) * DT
        self.theta += self.w * DT
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

    def pose(self):
        return np.array([self.x, self.y, self.theta])


def ray_segment_intersection(x, y, dx, dy, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    rx, ry = dx, dy
    sx, sy = x2 - x1, y2 - y1
    det = rx * sy - ry * sx
    if abs(det) < 1e-6:
        return None
    t = ((x1 - x) * sy - (y1 - y) * sx) / det
    u = ((x1 - x) * ry - (y1 - y) * rx) / det
    if t >= 0 and 0 <= u <= 1:
        return t
    return None


def lidar_scan(field, pose):
    x, y, theta = pose
    angles = np.linspace(-FOV / 2, FOV / 2, NUM_BEAMS)
    ranges = np.full(NUM_BEAMS, MAX_RANGE)

    for i, a in enumerate(angles):
        ang = theta + a
        dx, dy = np.cos(ang), np.sin(ang)
        dmin = MAX_RANGE
        for seg in field.lines:
            d = ray_segment_intersection(x, y, dx, dy, *seg)
            if d is not None and d < dmin:
                dmin = d
        ranges[i] = dmin + np.random.randn() * LIDAR_NOISE

    return ranges, angles


def low_variance_sampler(X_bar, W):
    M = len(X_bar)
    X_new = []
    r = np.random.uniform(0, 1.0 / M)
    c = W[0]
    i = 0
    for m in range(M):
        s = r + m * (1.0 / M)
        while s > c:
            i += 1
            c += W[i]
        X_new.append(X_bar[i])
    return np.array(X_new)


def particle_filter(X_prev, u_t, o_t, field):
    X_bar = []
    Wts = []

    for m in range(len(X_prev)):
        v, w = u_t
        x_prev = X_prev[m]

        noise = np.random.multivariate_normal([0, 0, 0], MOTION_NOISE)

        x = x_prev[0] + v * np.cos(x_prev[2]) * DT + noise[0]
        y = x_prev[1] + v * np.sin(x_prev[2]) * DT + noise[1]
        th = x_prev[2] + w * DT + noise[2]
        th = (th + np.pi) % (2 * np.pi) - np.pi

        x_t = np.array([x, y, th])

        z_pred, _ = lidar_scan(field, x_t)
        error = o_t - z_pred
        w_t = np.exp(-0.5 * np.sum(error**2) / (LIDAR_NOISE**2))

        X_bar.append(x_t)
        Wts.append(w_t)

    Wts = np.array(Wts)
    s = np.sum(Wts)
    if s <= 0 or not np.isfinite(s):
        Wts = np.ones_like(Wts) / len(Wts)
    else:
        Wts /= s

    X_resampled = low_variance_sampler(X_bar, Wts)

    M = len(X_resampled)
    K = int(RANDOM_PARTICLE_RATIO * M)

    for _ in range(K):
        idx = np.random.randint(M)
        X_resampled[idx, 0] = np.random.uniform(-FIELD_LENGTH / 2, FIELD_LENGTH / 2)
        X_resampled[idx, 1] = np.random.uniform(-FIELD_WIDTH / 2, FIELD_WIDTH / 2)
        X_resampled[idx, 2] = np.random.uniform(-np.pi, np.pi)

    return X_resampled


def bresenham(x0, y0, x1, y1):
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return cells


class Simulator:
    def __init__(self):
        self.field = SoccerField()
        self.robot = Robot()

        self.particles = np.zeros((NUM_PARTICLES, 3))
        self.particles[:, 0] = np.random.uniform(-FIELD_LENGTH / 2, FIELD_LENGTH / 2, NUM_PARTICLES)
        self.particles[:, 1] = np.random.uniform(-FIELD_WIDTH / 2, FIELD_WIDTH / 2, NUM_PARTICLES)
        self.particles[:, 2] = np.random.uniform(-np.pi, np.pi, NUM_PARTICLES)

        self.grid_w = int(np.ceil(FIELD_LENGTH / GRID_RES))
        self.grid_h = int(np.ceil(FIELD_WIDTH / GRID_RES))
        self.occ_grid = np.zeros((self.grid_w, self.grid_h), dtype=np.float32)

        self.fig1, self.ax1 = plt.subplots()
        self.fig2, self.ax2 = plt.subplots()

        self.fig1.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig1.canvas.mpl_connect("key_release_event", self.on_key_release)
        self.keys = set()

    def world_to_grid(self, x, y):
        gx = int((x + FIELD_LENGTH / 2) / GRID_RES)
        gy = int((y + FIELD_WIDTH / 2) / GRID_RES)
        if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
            return gx, gy
        return None

    def update_occupancy_grid(self, z_t, angles):
        rx, ry, rth = self.robot.pose()
        start = self.world_to_grid(rx, ry)
        if start is None:
            return

        for r, a in zip(z_t, angles):
            if not np.isfinite(r) or r <= 0.0 or r >= MAX_RANGE:
                continue

            ex = rx + r * np.cos(rth + a)
            ey = ry + r * np.sin(rth + a)

            end = self.world_to_grid(ex, ey)
            if end is None:
                continue

            cells = bresenham(start[0], start[1], end[0], end[1])
            if len(cells) < 2:
                continue

            for c in cells[:-1]:
                self.occ_grid[c] += LOG_ODD_FREE

            self.occ_grid[cells[-1]] += LOG_ODD_OCC

        np.clip(self.occ_grid, LOG_ODD_MIN, LOG_ODD_MAX, out=self.occ_grid)

    def on_key_press(self, event):
        self.keys.add(event.key)
        if "up" in self.keys:
            self.robot.v = V
        if "down" in self.keys:
            self.robot.v = -V
        if "left" in self.keys:
            self.robot.w = W
        if "right" in self.keys:
            self.robot.w = -W
        if event.key == "escape":
            plt.close("all")

    def on_key_release(self, event):
        if event.key in self.keys:
            self.keys.remove(event.key)
        if not ({"up", "down"} & self.keys):
            self.robot.v = 0.0
        if not ({"left", "right"} & self.keys):
            self.robot.w = 0.0

    def update(self):
        self.robot.step()

        z_t, angles = lidar_scan(self.field, self.robot.pose())

        self.particles = particle_filter(
            self.particles,
            (self.robot.v, self.robot.w),
            z_t,
            self.field,
        )

        self.update_occupancy_grid(z_t, angles)

        self.ax1.clear()
        self.field.draw(self.ax1)
        body = Circle((self.robot.x, self.robot.y), 2.0, color="C1")
        self.ax1.add_patch(body)

        self.ax2.clear()

        prob = 1.0 - 1.0 / (1.0 + np.exp(self.occ_grid))
        self.ax2.imshow(
            prob.T,
            origin="lower",
            cmap="gray",
            extent=[
                -FIELD_LENGTH / 2,
                FIELD_LENGTH / 2,
                -FIELD_WIDTH / 2,
                FIELD_WIDTH / 2,
            ],
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )

        self.ax2.scatter(self.particles[:, 0], self.particles[:, 1], s=8, alpha=0.35)

        for r, a in zip(z_t, angles):
            x = self.robot.x + r * np.cos(self.robot.theta + a)
            y = self.robot.y + r * np.sin(self.robot.theta + a)
            self.ax2.plot([self.robot.x, x], [self.robot.y, y], color="C0", alpha=0.2)

        self.ax2.set_aspect("equal")
        self.ax2.set_xlim(-FIELD_LENGTH / 2, FIELD_LENGTH / 2)
        self.ax2.set_ylim(-FIELD_WIDTH / 2, FIELD_WIDTH / 2)
        self.ax2.set_title("Particle Filter + Occupancy Grid")

        self.fig1.canvas.draw_idle()
        self.fig2.canvas.draw_idle()

    def run(self):
        plt.ion()
        while plt.fignum_exists(self.fig1.number):
            self.update()
            plt.pause(DT)


if __name__ == "__main__":
    Simulator().run()
