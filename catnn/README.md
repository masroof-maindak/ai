# Setup

```
# Download
wget https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz

# Extract
tar xzvf images.tar.gz

# Keep images for only 5 cat classes
find images/* -type f ! -regex  '\(.*Abyssinian.*\|.*Bengal.*\|.*Bombay.*\|.*Egyptian_Mau.*\|.*Russian_Blue.*\)$' -delete

# Install Python packages
uv sync
```

# Usage

```
uv run main.py
```

# Acknowledgements

- [Oxford-IIIT Pets Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
