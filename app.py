import streamlit as st
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from model import DigitGenerator  # from train script

st.title("MNIST Digit Generator")
digit = st.selectbox("Choose a digit (0-9)", list(range(10)))

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitGenerator()
model.load_state_dict(torch.load("digit_generator.pth", map_location=device))
model.eval()

# Generate images
noise = torch.randn(5, 100)
labels = torch.tensor([digit]*5)
with torch.no_grad():
    images = model(noise, labels).squeeze(1)

# Display images
grid = make_grid(images, nrow=5, normalize=True)
fig, ax = plt.subplots()
ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
ax.axis('off')
st.pyplot(fig)
