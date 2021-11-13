# DDPW Documentation

This is the source for the [DDPW
Documentation](http://ddpw.projects-tvs.surge.sh) written in Python 3.8 with
[Sphinx](https://www.sphinx-doc.org/en/master/).

## Set up

```bash
# with conda
conda env create --file environment.yaml
conda activate ddpw-docs

# with pip
pip install -r requirements.txt
```

**Freeze environment**

```bash
sh freeze.sh

# alternatively
conda export env > environment.yaml
pip list --format=freeze > requirements.txt
```

**Update environent**

```bash
conda env update --file environment.yaml
```

## Publish manually

```bash
sh publish.sh
```