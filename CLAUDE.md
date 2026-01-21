# Claude Code Project Configuration

## Python File Header Convention

When creating new Python files, always include:

1. **Shebang line** (first line):
   ```python
   #!/usr/bin/env python3
   ```

2. **Docstring with metadata** (immediately after shebang):
   ```python
   """
   Brief description of what the file does.

   Created: YYYY-MM-DD
   Author: Wataru Fukuda
   """
   ```

## Project Structure

```
watfIGA/
├── watfIGA/           # Core library
│   ├── geometry/      # NURBS curves/surfaces, B-spline basis
│   ├── discretization/# Knot vectors, Bezier extraction, mesh building
│   ├── solver/        # PDE solvers (Poisson, elasticity, plates)
│   ├── quadrature/    # Gauss-Legendre integration
│   ├── postprocess/   # Solution sampling, VTK export
│   ├── visualization/ # XMF2 export for ParaView
│   └── io/            # CAD import, configuration
├── examples/
│   └── src/           # Example scripts (executable)
├── tests/             # Unit tests
└── docs/              # Documentation
```

## Code Style

- Use x-fastest ordering for multi-dimensional arrays: `index = j * n_xi + i`
- Big-endian format for binary files ('>f8' for float64, '>i4' for int32)
- Prefer editing existing files over creating new ones

## Running Examples

Examples in `examples/src/` are executable:
```bash
./examples/src/square_mesh.py
./examples/src/nurbs_poisson_2d.py
```

## Visualization

- Export to XMF2 format for ParaView visualization
- Use `generate_xmf(mesh_dir, mode="cps")` for control point grid
- Use `generate_xmf(mesh_dir, mode="mesh")` for IGA mesh
