# recurrence2d

Generalised recurrence plots for spatial data: the 2-D extension of recurrence plots and recurrence
quantification analysis (Marwan, Kurths & Saparin 2007, *Physics Letters A* 360, 545–551), with joint and
cross generalised plots for co-registered fields, spatial recurrence networks and sliding-window measure fields.

- `generalized/`: the C++ implementation and its test programs (`make` in that folder; see its README for the
  measures and equations).
- `python/`: the `recurrence2d` package, a pybind11 binding over the same C++ with OpenMP, plus raster I/O and
  EBSD loaders (`pip install -e python/`; the extension compiles `generalized/*.cpp`).

For recurrence plots of time series (the 1-D case, with delay embedding) see Barbara Reis's
[bmfreis/recurrence_cpp](https://github.com/bmfreis/recurrence_cpp), a separate project.

## License

MIT (`LICENSE`).
