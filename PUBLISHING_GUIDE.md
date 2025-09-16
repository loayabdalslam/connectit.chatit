# Publishing ConnectIT to PyPI

This guide walks you through the process of publishing the ConnectIT library to the Python Package Index (PyPI).

## Prerequisites

1. **Create PyPI Account**
   - Go to [https://pypi.org/account/register/](https://pypi.org/account/register/)
   - Create an account and verify your email

2. **Create TestPyPI Account** (recommended for testing)
   - Go to [https://test.pypi.org/account/register/](https://test.pypi.org/account/register/)
   - Create an account for testing uploads

3. **Install Required Tools**
   ```bash
   pip install --upgrade pip
   pip install --upgrade build twine
   ```

## Step-by-Step Publishing Process

### 1. Prepare Your Package

Ensure your package is ready:
- ✅ `pyproject.toml` is properly configured
- ✅ `LICENSE` file exists
- ✅ `README.md` is comprehensive
- ✅ `MANIFEST.in` includes all necessary files
- ✅ Version number is correct in `pyproject.toml`

### 2. Clean Previous Builds

```bash
# Remove old build artifacts
rm -rf dist/
rm -rf build/
rm -rf *.egg-info/
```

### 3. Build the Package

```bash
# Build source distribution and wheel
python -m build
```

This creates:
- `dist/connectit-0.1.0.tar.gz` (source distribution)
- `dist/connectit-0.1.0-py3-none-any.whl` (wheel)

### 4. Test Upload to TestPyPI (Recommended)

```bash
# Upload to TestPyPI first
python -m twine upload --repository testpypi dist/*
```

You'll be prompted for:
- Username: Your TestPyPI username
- Password: Your TestPyPI password (or API token)

### 5. Test Installation from TestPyPI

```bash
# Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ connectit
```

### 6. Upload to Production PyPI

Once testing is successful:

```bash
# Upload to production PyPI
python -m twine upload dist/*
```

You'll be prompted for:
- Username: Your PyPI username
- Password: Your PyPI password (or API token)

## Using API Tokens (Recommended)

For better security, use API tokens instead of passwords:

1. **Generate API Token**
   - Go to PyPI Account Settings → API tokens
   - Create a new token with appropriate scope
   - Copy the token (starts with `pypi-`)

2. **Configure Credentials**
   Create `~/.pypirc`:
   ```ini
   [distutils]
   index-servers =
       pypi
       testpypi

   [pypi]
   username = __token__
   password = pypi-your-api-token-here

   [testpypi]
   repository = https://test.pypi.org/legacy/
   username = __token__
   password = pypi-your-testpypi-token-here
   ```

## Version Management

### Updating Versions

1. **Update version in `pyproject.toml`**:
   ```toml
   version = "0.1.1"  # Increment version
   ```

2. **Follow Semantic Versioning**:
   - `0.1.0` → `0.1.1` (patch: bug fixes)
   - `0.1.0` → `0.2.0` (minor: new features)
   - `0.1.0` → `1.0.0` (major: breaking changes)

### Re-publishing

```bash
# Clean, build, and upload new version
rm -rf dist/ build/ *.egg-info/
python -m build
python -m twine upload dist/*
```

## Installation Commands for Users

Once published, users can install ConnectIT:

```bash
# Basic installation
pip install connectit

# With optional dependencies
pip install connectit[hf]        # Hugging Face support
pip install connectit[torch]     # PyTorch support
pip install connectit[all]       # All optional dependencies
```

## Troubleshooting

### Common Issues

1. **"File already exists" error**
   - You cannot upload the same version twice
   - Increment the version number in `pyproject.toml`

2. **Missing files in package**
   - Check `MANIFEST.in` includes all necessary files
   - Verify with `python -m build --sdist` and inspect the `.tar.gz`

3. **Import errors after installation**
   - Ensure `__init__.py` files exist in all packages
   - Check that dependencies are correctly specified

4. **Authentication errors**
   - Verify your PyPI credentials
   - Consider using API tokens instead of passwords

### Checking Package Contents

```bash
# List contents of built package
tar -tzf dist/connectit-0.1.0.tar.gz
```

## Post-Publication

1. **Test Installation**
   ```bash
   pip install connectit
   python -c "import connectit; print('Success!')"
   ```

2. **Update Documentation**
   - Add installation instructions to README
   - Update version badges if using any

3. **Tag Release** (if using Git)
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

## Security Notes

- ⚠️ Never commit API tokens to version control
- ⚠️ Use API tokens instead of passwords when possible
- ⚠️ Test on TestPyPI before production uploads
- ⚠️ Keep your PyPI account secure with 2FA

## Contact

For commercial licensing or questions about ConnectIT:
📧 loaiabdalslam@gmail.com