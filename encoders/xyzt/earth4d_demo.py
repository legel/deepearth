# https://github.com/legel/deepearth
from deepearth.encoders.xyzt.earth4d import Earth4D

world_model = Earth4D()
embeddings = world_model(
    # Bletchley Park (Turing breaks Enigma, 1941)
    (51.9976, -0.7416, 110, "1941-06-01 09:00 GMT"),
    # Carnegie Mellon (Hinton invents Boltzmann Machines, 1985)
    (40.4433, -79.9436, 270, "1985-01-15 10:00 ET"),
    # CERN (Berners-Lee invents WWW, 1989)
    (46.2330, 6.0557, 430, "1989-03-12 10:00 CET"),
    # Mila, Quebec (World Modeling Workshop 2026)
    (45.5308, -73.6128, 63, "2026-02-04 11:00 ET"),
)
# embeddings.shape: [4, 192] -- trainable space-time features
