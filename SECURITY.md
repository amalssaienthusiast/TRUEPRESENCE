# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of TruePresence seriously. If you discover a security vulnerability, please follow the responsible disclosure process outlined below.

### How to Report

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please report security vulnerabilities by emailing the project maintainers directly. Include:

1. **Description** of the vulnerability
2. **Steps to reproduce** the issue
3. **Potential impact** of the vulnerability
4. **Suggested fix** (if you have one)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Fix Timeline**: Depends on severity (see below)

| Severity | Fix Timeline |
|----------|--------------|
| Critical | 24-48 hours |
| High | 1 week |
| Medium | 2-4 weeks |
| Low | Next release |

### What to Expect

1. Confirmation of receipt
2. Assessment of the vulnerability
3. Regular updates on progress
4. Credit in the security advisory (if desired)

---

## Security Best Practices

When deploying TruePresence, follow these security recommendations:

### Deployment Security

#### 1. Network Security
```python
# Production Flask configuration
app.run(
    host='127.0.0.1',  # Don't expose to 0.0.0.0 in production
    debug=False,       # Disable debug mode
    threaded=True
)
```

- Use a reverse proxy (nginx, Apache) for production
- Enable HTTPS/TLS encryption
- Restrict network access to trusted IPs

#### 2. Authentication
The current version does not include authentication. For production use:

```python
# Example: Add basic authentication
from functools import wraps
from flask import request, Response

def check_auth(username, password):
    return username == 'admin' and password == 'secure_password'

def authenticate():
    return Response('Access Denied', 401, 
                    {'WWW-Authenticate': 'Basic realm="TruePresence"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated
```

#### 3. Database Security
- Store the SQLite database outside the web root
- Regular database backups
- Set appropriate file permissions (600)

```bash
# Restrict database access
chmod 600 attendance.db
```

---

## Known Security Considerations

### Anti-Spoofing Limitations

| Attack Vector | Mitigation | Limitation |
|---------------|------------|------------|
| Static photos | Texture analysis, blink detection | High-quality prints may bypass |
| Video playback | Motion analysis | High FPS recordings may bypass |
| 3D masks | Texture + motion analysis | High-quality masks may bypass |

**Recommendation**: For high-security deployments, combine with:
- Multi-factor authentication
- Depth-sensing cameras (Intel RealSense, etc.)
- Human oversight for suspicious activity

### Data Privacy

Face data and biometric information are sensitive. Ensure compliance with:

- **GDPR** (European Union)
- **CCPA** (California)
- **Local data protection laws**

Recommendations:
- Inform users about data collection
- Obtain consent before registration
- Provide data deletion mechanisms
- Encrypt stored face features

### Process Execution

The Flask app executes Python scripts via `subprocess`:

```python
# Current implementation (be aware of risks)
subprocess.Popen([sys.executable, script_path], ...)
```

This is safe for the intended scripts but:
- Never allow user-supplied script names
- Validate all input paths
- Run with minimal privileges

---

## Security Checklist for Deployment

- [ ] Disable Flask debug mode
- [ ] Use HTTPS in production
- [ ] Implement authentication
- [ ] Set secure file permissions
- [ ] Regular security updates
- [ ] Monitor for unusual activity
- [ ] Backup database regularly
- [ ] Review access logs
- [ ] Restrict camera access
- [ ] Secure dlib model files

---

## Vulnerability Disclosure Hall of Fame

We appreciate security researchers who help keep TruePresence secure.

*No vulnerabilities reported yet.*

---

## Contact

For security concerns, please contact the project maintainers.

---

*Last updated: January 2025*
