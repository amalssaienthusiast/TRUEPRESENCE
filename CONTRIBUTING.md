# Contributing to TruePresence

First off, thank you for considering contributing to TruePresence! It's people like you that make TruePresence such a great tool for secure attendance management.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

---

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

### Our Standards

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

---

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/True_Presence.git
   cd True_Presence
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## How Can I Contribute?

### 1. Reporting Bugs

Found a bug? Please open an issue with:

- A clear, descriptive title
- Steps to reproduce the bug
- Expected behavior vs. actual behavior
- Your environment (OS, Python version, etc.)
- Screenshots if applicable

### 2. Suggesting Enhancements

Have an idea? Open an issue with:

- A clear description of the enhancement
- Why this would be useful
- Any examples or mockups

### 3. Code Contributions

We welcome code contributions! Priority areas include:

| Area | Description | Priority |
|------|-------------|----------|
| Unit Tests | Add pytest tests for core functions | High |
| Anti-Spoofing | Implement additional detection methods | High |
| Multi-camera | Support for multiple camera inputs | Medium |
| UI/UX | Mobile-responsive improvements | Medium |
| Documentation | Translations, tutorials | Medium |
| Performance | Optimization for edge devices | Low |

---

## Development Setup

### Prerequisites

- Python 3.8+
- pip
- CMake (for dlib)
- A webcam for testing

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/True_Presence.git
cd True_Presence

# Create virtual environment
python3 -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Check code style
flake8 *.py flask_app/*.py
```

---

## Pull Request Process

### Before Submitting

1. **Test your changes** thoroughly
2. **Update documentation** if needed
3. **Follow style guidelines** (see below)
4. **Write clear commit messages**

### Submitting

1. Push your branch to your fork
2. Open a Pull Request against `main`
3. Fill out the PR template completely
4. Link any related issues

### PR Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged

### Commit Message Format

```
type(scope): brief description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(anti-spoof): add depth-based liveness detection

fix(camera): handle camera initialization timeout

docs(readme): add installation guide for Windows
```

---

## Style Guidelines

### Python Code Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with these specifics:

```python
# Use meaningful variable names
face_descriptor = compute_face_encoding(image)  # ✓
fd = cfe(img)  # ✗

# Add docstrings to functions
def calculate_ear(eye_points):
    """
    Calculate the Eye Aspect Ratio (EAR).
    
    Args:
        eye_points: List of 6 (x, y) landmark coordinates
        
    Returns:
        float: The calculated EAR value
    """
    pass

# Use type hints for new code
def detect_blink(self, shape: np.ndarray) -> tuple[bool, float]:
    pass
```

### File Organization

```python
# Standard library imports
import os
import time

# Third-party imports
import cv2
import numpy as np

# Local imports
from .utils import helper_function
```

### HTML/CSS/JavaScript

- Use 4-space indentation
- Use semantic HTML5 elements
- Follow BEM naming convention for CSS classes
- Use modern ES6+ JavaScript features

---

## Reporting Bugs

### Bug Report Template

```markdown
## Bug Description
A clear description of the bug.

## Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. See error

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Environment
- OS: [e.g., macOS 13.0]
- Python: [e.g., 3.10.0]
- OpenCV: [e.g., 4.7.0]
- dlib: [e.g., 19.24.0]

## Screenshots
If applicable, add screenshots.

## Additional Context
Any other relevant information.
```

---

## Suggesting Enhancements

### Enhancement Template

```markdown
## Feature Description
A clear description of the enhancement.

## Use Case
Why is this feature needed? What problem does it solve?

## Proposed Solution
How might this be implemented?

## Alternatives Considered
Any alternative solutions you've thought about.

## Additional Context
Mockups, examples, or references.
```

---

## Questions?

Feel free to open an issue with the `question` label or reach out to the maintainers.

---

Thank you for contributing to TruePresence! 🎉
