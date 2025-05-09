import numpy as np

def sample(texture, u, v):
    # Contraindre u et v dans l'intervalle [0, 1]
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)

    u = int(u * (texture.shape[1] - 1))  
    v = int((1 - v) * (texture.shape[0] - 1))  
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

    def VertexShader(self, vertex, data) :
        outputVertex = np.zeros((14))

        x = vertex[0]
        y = vertex[1]
        z = vertex[2]
        w = 1.0

        vec = np.array([[x],[y],[z],[w]])

        vec = np.matmul(data['projMatrix'],np.matmul(data['viewMatrix'],vec))

        outputVertex[0] = vec[0]/vec[3]
        outputVertex[1] = vec[1]/vec[3]
        outputVertex[2] = vec[2]/vec[3]

        outputVertex[3] = vertex[3]
        outputVertex[4] = vertex[4]
        outputVertex[5] = vertex[5]

        outputVertex[6] = data['cameraPosition'][0] - vertex[0]
        outputVertex[7] = data['cameraPosition'][1] - vertex[1]
        outputVertex[8] = data['cameraPosition'][2] - vertex[2]

        outputVertex[9] = data['lightPosition'][0] - vertex[0]
        outputVertex[10] = data['lightPosition'][1] - vertex[1]
        outputVertex[11] = data['lightPosition'][2] - vertex[2]

        outputVertex[12] = vertex[6]
        outputVertex[13] = vertex[7]

        return outputVertex


    def Rasterizer(self, v0, v1, v2) :
        fragments = []

        #culling back face
        area = edgeSide(v0,v1,v2)
        if area < 0 :
            return fragments
        
        
        #AABBox computation
        #compute vertex coordinates in screen space
        v0_image = np.array([0,0])
        v0_image[0] = (v0[0]+1.0)/2.0 * self.width 
        v0_image[1] = ((v0[1]+1.0)/2.0) * self.height 

        v1_image = np.array([0,0])
        v1_image[0] = (v1[0]+1.0)/2.0 * self.width 
        v1_image[1] = ((v1[1]+1.0)/2.0) * self.height 

        v2_image = np.array([0,0])
        v2_image[0] = (v2[0]+1.0)/2.0 * self.width 
        v2_image[1] = (v2[1]+1.0)/2.0 * self.height 

        #compute the two point forming the AABBox
        A = np.min(np.array([v0_image,v1_image,v2_image]), axis = 0)
        B = np.max(np.array([v0_image,v1_image,v2_image]), axis = 0)

        #cliping the bounding box with the borders of the image
        max_image = np.array([self.width-1,self.height-1])
        min_image = np.array([0.0,0.0])

        A  = np.max(np.array([A,min_image]),axis = 0)
        B  = np.min(np.array([B,max_image]),axis = 0)
        
        #cast bounding box to int
        A = A.astype(int)
        B = B.astype(int)
        #Compensate rounding of int cast
        B = B + 1

        #for each pixel in the bounding box
        for j in range(A[1], B[1]) : 
           for i in range(A[0], B[0]) :
                x = (i+0.5)/self.width * 2.0 - 1 
                y = (j+0.5)/self.height * 2.0 - 1

                p = np.array([x,y])
                
                area0 = edgeSide(p,v0,v1)
                area1 = edgeSide(p,v1,v2)
                area2 = edgeSide(p,v2,v0)

                #test if p is inside the triangle
                if (area0 >= 0 and area1 >= 0 and area2 >= 0) : 
                    
                    #Computing 2d barricentric coordinates
                    lambda0 = area1/area
                    lambda1 = area2/area
                    lambda2 = area0/area
                    
                    #one_over_z = lambda0 * 1/v0[2] + lambda1 * 1/v1[2] + lambda2 * 1/v2[2]
                    #z = 1/one_over_z
                    z = lambda0 * v0[2] + lambda1 * v1[2] + lambda2 * v2[2]
                    p = np.array([x,y,z])
                    l = v0.shape[0]
                    #interpolating
                    interpolated_data = v0[3:l] * lambda0 + v1[3:l] * lambda1 + v2[3:l] * lambda2
                    
                    #Emiting Fragment
                    fragments.append(Fragment(i,j,z, interpolated_data))

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

    def RasterizerMSAAx8(self, v0, v1, v2):
        fragments = []
        area = edgeSide(v0, v1, v2)
        if area < 0:
            return fragments
        sample_positions = [
            (0.0625, 0.1875), (0.3125, 0.4375), 
            (0.6875, 0.0625), (0.9375, 0.3125),
            (0.1875, 0.8125), (0.4375, 0.9375),
            (0.8125, 0.5625), (0.5625, 0.6875)
        ]
    
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


    def fragmentShader(self,fragment,data):
        N = fragment.interpolated_data[0:3]
        N = N/np.linalg.norm(N)
        V = fragment.interpolated_data[3:6]
        V = V/np.linalg.norm(V)
        L = fragment.interpolated_data[6:9]
        L = L/np.linalg.norm(L)

        R = 2 * np.dot(L,N) * N  -L

        

        ambient = 1.0
        diffuse = max(np.dot(N,L),0)
        specular = np.power(max(np.dot(R,V),0.0),64) 

        ka = 0.1
        kd = 0.9
        ks = 0.3
        phong = ka * ambient + kd * diffuse + ks * specular
        phong = np.ceil(phong*4 +1 )/6.0

        texture = sample(data['texture'], fragment.interpolated_data[9], fragment.interpolated_data[10])


        color = np.array([phong,phong,phong]) * texture

        fragment.output = color

    def draw(self, vertices, triangles, data, MSAA=False, MSAA8x=False):
        self.image = np.zeros((self.height, self.width, 3))
        
        # Préparer les buffers selon le mode
        if MSAA8x:
            # Mode MSAA 8x
            self.msaaBuffer = np.zeros((self.height, self.width, 8, 3))
            self.msaaDepth = np.ones((self.height, self.width, 8))
            self.coverage = np.zeros((self.height, self.width), dtype=int)
        elif MSAA:
            # Mode MSAA 4x
            self.msaaBuffer = np.zeros((self.height, self.width, self.samples, 3))
            self.msaaDepth = np.ones((self.height, self.width, self.samples))
            self.coverage = np.zeros((self.height, self.width), dtype=int)
        else:
            # Mode normal 
            self.singleBuffer = np.zeros((self.height, self.width, 3))
            self.singleDepth = np.ones((self.height, self.width))
        
        self.newVertices = np.zeros((vertices.shape[0], 14))
        for i in range(vertices.shape[0]):
            self.newVertices[i] = self.VertexShader(vertices[i], data)
        
        fragments = []
        for i in triangles:
            if MSAA8x:
                fragments.extend(self.RasterizerMSAAx8(self.newVertices[i[0]], self.newVertices[i[1]], self.newVertices[i[2]]))
            elif MSAA:
                fragments.extend(self.RasterizerMSAA(self.newVertices[i[0]], self.newVertices[i[1]], self.newVertices[i[2]]))
            else:
                fragments.extend(self.Rasterizer(self.newVertices[i[0]], self.newVertices[i[1]], self.newVertices[i[2]]))
        
        for f in fragments:
            self.fragmentShader(f, data)
            
            if MSAA or MSAA8x:
                # Mode MSAA
                if f.sample_id is None:
                    f.sample_id = 0
                if self.msaaDepth[f.y, f.x, f.sample_id] > f.depth:
                    self.msaaDepth[f.y, f.x, f.sample_id] = f.depth
                    self.msaaBuffer[f.y, f.x, f.sample_id] = f.output
                    self.coverage[f.y, f.x] += 1  
            else:
                if self.singleDepth[f.y, f.x] > f.depth:
                    self.singleDepth[f.y, f.x] = f.depth
                    self.singleBuffer[f.y, f.x] = f.output
        
        if MSAA or MSAA8x:
            # Résolution MSAA
            if MSAA:
                samples_count = 4
            elif MSAA8x:
                samples_count = 8
            for y in range(self.height):
                for x in range(self.width):
                    if self.coverage[y, x] > 0:
                        self.image[y, x] = np.sum(self.msaaBuffer[y, x], axis=0) / samples_count
        else:
            self.image = self.singleBuffer.copy()