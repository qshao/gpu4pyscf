# Resident Taxol DF J/K block-size sweep

| Auxiliary block | Median J/K, s | Paired J/K speed ratio vs 256 | Estimated unpack + transform, GiB |
| ---: | ---: | ---: | ---: |
| 128 | 13.873715 | 0.9927 | 1.445 |
| 256 | 13.772836 | 1.0000 | 2.889 |
| 512 | 13.879403 | 0.9924 | 5.779 |
| 1024 | 14.132802 | 0.9744 | 11.558 |

Speed ratio is the reciprocal of the median of three candidate/control time ratios. The 256 median uses all nine paired control samples; warmups are excluded. Each of 22 calls passes the finite full-J/K component error gate of absolute 1e-9, relative zero.

Fixed-density J/K only: no SCF, forces or full-workflow gain is measured. Workspace is an estimate of dense AO unpack plus one occupied transform buffer, not measured total RAM; it excludes persistent CDERI, outputs, factors, allocator reservation and native-library workspaces. Three timing pairs do not establish a hardware-wide guarantee. The audit rechecks recorded finite full-matrix maximum errors; J/K matrices were not persisted for a separate numerical recomputation.

Source, native libraries, helpers, inputs, original scientific settings and saved-orbital CPU comparison were independently rechecked.
