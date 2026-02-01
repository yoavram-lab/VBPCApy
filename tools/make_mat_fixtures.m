% tools/make_mat_fixtures.m
%
% Generates MATLAB-compatible .mat fixtures for the Python test suite:
%   tests/data/legacy_pca_full_dense.mat
%   tests/data/legacy_pca_full_missing.mat
%
% If you do not have a MATLAB license, you can run this script using GNU
% Octave (https://www.gnu.org/software/octave/). It should be fully
% compatible.

% -----------------------------
% ADD YOUR LEGACY MATLAB VBPCA CODE PATH HERE
% can be downloaded from: https://users.ics.aalto.fi/alexilin/software/
% -----------------------------
% This must be the folder that contains pca_full.m (or a parent directory).
addpath("/Users/josh/Documents/VBPCA");  

% Confirm pca_full is found
if exist("pca_full", "file") ~= 2
  error("Could not find pca_full.m on the Octave path. Edit addpath(...) above.");
end

% Output directory: tests/data
outdir = fullfile(pwd, "tests", "data");
if exist(outdir, "dir") ~= 7
  mkdir(outdir);
end

% Use -v7 for broad compatibility with scipy.io.loadmat
save_format = "-v7";

% -------------------------------------------------------------------------
% Helper: build options struct (matches what Python tests pass)
% -------------------------------------------------------------------------
function opts = _make_opts(maxiters, bias, uniquesv)
  opts = struct();
  opts.init = "random";
  opts.maxiters = maxiters;
  opts.bias = bias;
  opts.uniquesv = uniquesv;
  opts.autosave = 0;
  opts.filename = "pca_f_autosave";
  opts.minangle = 1e-8;
  opts.algorithm = "vb";
  opts.niter_broadprior = 100;
  opts.earlystop = 0;
  opts.rmsstop = [100, 1e-4, 1e-3];
  opts.cfstop = [];
  opts.verbose = 0;
  opts.xprobe = [];
  opts.rotate2pca = 1;
  opts.display = 0;
end

% -------------------------------------------------------------------------
% 1) Dense fixture
% -------------------------------------------------------------------------
% Deterministic RNG (Octave-compatible)
rand("state", 1);
randn("state", 1);

n_features = 6;
n_samples = 10;
x = randn(n_features, n_samples);
k = 3;

opts = _make_opts(200, 1, 0);

result = pca_full(x, k, opts);

% Store a few fields that the Python test tries to read from mat_res
% (It uses these only to populate args to Python pca_full.)
result.maxiters = opts.maxiters;
result.bias = opts.bias;
result.uniquesv = opts.uniquesv;

dense_path = fullfile(outdir, "legacy_pca_full_dense.mat");
save(dense_path, save_format, "x", "k", "result");
fprintf("Wrote %s\n", dense_path);

% -------------------------------------------------------------------------
% 2) Missing-data fixture
% -------------------------------------------------------------------------
rand("state", 2);
randn("state", 2);

x = randn(n_features, n_samples);
mask = rand(n_features, n_samples) < 0.2;
x(mask) = NaN;
k = 3;

opts = _make_opts(300, 1, 1);

result = pca_full(x, k, opts);
result.maxiters = opts.maxiters;
result.bias = opts.bias;
result.uniquesv = opts.uniquesv;

missing_path = fullfile(outdir, "legacy_pca_full_missing.mat");
save(missing_path, save_format, "x", "k", "result");
fprintf("Wrote %s\n", missing_path);

fprintf("Done.\n");