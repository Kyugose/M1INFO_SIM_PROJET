import numpy as np

def sample(texture, u, v):
    u = int(u * texture.shape[0])
    v = int((1 - v) * texture.shape[1])
    return texture[v, u] / 255.0

class Fragment:
    def __init__(self, x: int, y: int, depth: float, interpolated_data, sample_id=0):
        self.x = x
        self.y = y
        self.depth = depth
        self.interpolated_data = interpolated_data
        self.sample_id = sample_id
        self.output = []

def edgeSide(p, v0, v1):
    return (p[0] - v0[0]) * (v1[1] - v0[1]) - (p[1] - v0[1]) * (v1[0] - v0[0])

class GraphicPipeline:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.image = np.zeros((height, width, 3))
        self.samples = 4
        self.msaaBuffer = np.zeros((height, width, self.samples, 3))
        self.msaaDepth = np.ones((height, width, self.samples))
        self.coverage = np.zeros((height, width), dtype=int)

    def VertexShader(self, vertex, data):
        outputVertex = np.zeros((14))
        x = vertex[0]
        y = vertex[1]
        z = vertex[2]
        w = 1.0
        vec = np.array([[x], [y], [z], [w]])
        vec = np.matmul(data['projMatrix'], np.matmul(data['viewMatrix'], vec))
        outputVertex[0] = vec[0] / vec[3]
        outputVertex[1] = vec[1] / vec[3]
        outputVertex[2] = vec[2] / vec[3]
        outputVertex[3:6] = vertex[3:6]
        outputVertex[6:9] = data['cameraPosition'] - vertex[0:3]
        outputVertex[9:12] = data['lightPosition'] - vertex[0:3]
        outputVertex[12:14] = vertex[6:8]
        return outputVertex

    def Rasterizer(self, v0, v1, v2):
        fragments = []
        area = edgeSide(v0, v1, v2)
        if area < 0:
            return fragments
        verts = [v0, v1, v2]
        verts_image = [((v[0] + 1.0) / 2.0 * self.width, (v[1] + 1.0) / 2.0 * self.height) for v in verts]
        A = np.min(verts_image, axis=0).astype(int)
        B = np.max(verts_image, axis=0).astype(int) + 1
        for j in range(A[1], B[1]):
            for i in range(A[0], B[0]):
                x = (i + 0.5) / self.width * 2.0 - 1
                y = (j + 0.5) / self.height * 2.0 - 1
                p = np.array([x, y])
                area0 = edgeSide(p, v0, v1)
                area1 = edgeSide(p, v1, v2)
                area2 = edgeSide(p, v2, v0)
                if area0 >= 0 and area1 >= 0 and area2 >= 0:
                    lambda0 = area1 / area
                    lambda1 = area2 / area
                    lambda2 = area0 / area
                    z = lambda0 * v0[2] + lambda1 * v1[2] + lambda2 * v2[2]
                    interpolated_data = v0[3:] * lambda0 + v1[3:] * lambda1 + v2[3:] * lambda2
                    fragments.append(Fragment(i, j, z, interpolated_data))
        return fragments

    def RasterizerMSAA(self, v0, v1, v2):
        fragments = []
        area = edgeSide(v0, v1, v2)
        if area < 0:
            return fragments
        sample_positions = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
        verts_image = [((v[0] + 1) / 2 * self.width, (v[1] + 1) / 2 * self.height) for v in [v0, v1, v2]]
        A = np.min(verts_image, axis=0).astype(int)
        B = np.max(verts_image, axis=0).astype(int) + 1
        for y in range(A[1], B[1]):
            for x in range(A[0], B[0]):
                for s, (dx, dy) in enumerate(sample_positions):
                    px = ((x + dx) / self.width) * 2 - 1
                    py = ((y + dy) / self.height) * 2 - 1
                    p = np.array([px, py])
                    e0 = edgeSide(p, v0, v1)
                    e1 = edgeSide(p, v1, v2)
                    e2 = edgeSide(p, v2, v0)
                    if e0 >= 0 and e1 >= 0 and e2 >= 0:
                        l0, l1, l2 = e1 / area, e2 / area, e0 / area
                        z = l0 * v0[2] + l1 * v1[2] + l2 * v2[2]
                        interp_data = l0 * v0[3:] + l1 * v1[3:] + l2 * v2[3:]
                        fragments.append(Fragment(x, y, z, interp_data, s))
        return fragments

    def fragmentShader(self, fragment, data):
        N = fragment.interpolated_data[0:3]
        V = fragment.interpolated_data[3:6]
        L = fragment.interpolated_data[6:9]
        uv = fragment.interpolated_data[9:11]

        N_norm = np.linalg.norm(N)
        V_norm = np.linalg.norm(V)
        L_norm = np.linalg.norm(L)

        if N_norm > 0: N = N / N_norm
        if V_norm > 0: V = V / V_norm
        if L_norm > 0: L = L / L_norm

        R = 2 * np.dot(N, L) * N - L
        R_norm = np.linalg.norm(R)
        if R_norm > 0: R = R / R_norm

        ambient = 0.2
        diffuse = max(np.dot(N, L), 0.0)
        specular = np.power(max(np.dot(R, V), 0.0), 16)

        ka = 0.1
        kd = 0.8
        ks = 0.5

        light_intensity = ka * ambient + kd * diffuse + ks * specular
        tex = sample(data['texture'], uv[0], uv[1])

        fragment.output = np.clip(tex * light_intensity, 0, 1)

    def draw(self, vertices, triangles, data, MSAA=False, MSAA8x=False):
        self.image = np.zeros((self.height, self.width, 3))
        self.msaaBuffer = np.zeros((self.height, self.width, self.samples, 3))
        self.msaaDepth = np.ones((self.height, self.width, self.samples))
        self.coverage = np.zeros((self.height, self.width), dtype=int)
        self.newVertices = np.zeros((vertices.shape[0], 14))
        for i in range(vertices.shape[0]):
            self.newVertices[i] = self.VertexShader(vertices[i], data)
        fragments = []
        for i in triangles:
            v0 = self.newVertices[i[0]]
            v1 = self.newVertices[i[1]]
            v2 = self.newVertices[i[2]]
            if MSAA:
                fragments.extend(self.RasterizerMSAA(v0, v1, v2))
            else:
                fragments.extend(self.Rasterizer(v0, v1, v2))
        for f in fragments:
            self.fragmentShader(f, data)
            if f.sample_id is None:
                f.sample_id = 0
            if self.msaaDepth[f.y, f.x, f.sample_id] > f.depth:
                self.msaaDepth[f.y, f.x, f.sample_id] = f.depth
                self.msaaBuffer[f.y, f.x, f.sample_id] = f.output
                self.coverage[f.y, f.x] += 1
        for y in range(self.height):
            for x in range(self.width):
                count = self.coverage[y, x]
                if count > 0:
                    self.image[y, x] = np.sum(self.msaaBuffer[y, x], axis=0) / count