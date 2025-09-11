# Test set creation

This script is for sampling the 100 images used as test set. Sampling is done
with uniform spacing over distance travelled to attain a better sample diversity
and complete coverage of the area than random sampling.

Make sure the dataset is loaded under `/workspaces/collaborative-scene-graphs/data/radar-robotcar` and the paths are correctly set in `./sample_by_distance.py`.
Then create an empty directory `./test/` to store the test files.

```bash
python3 sample_by_distance.py
```

The test files are now in `test/`. A plot of the trajectory and the samples will be written to `trajectory.png`:

<img src="./trajectory.png" alt="Trajectory Plot" width="400">