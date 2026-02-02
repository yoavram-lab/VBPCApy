% tools/make_mat_fixtures.m
%
% Generates MATLAB-compatible .mat fixtures for the Python test suite:
%   tests/data/legacy_pca_full_dense.mat
%   tests/data/legacy_pca_full_missing.mat
%   tests/data/legacy_rotate_to_pca.mat        <-- NEW
%
% Run in MATLAB or GNU Octave.

% -----------------------------
% ADD YOUR LEGACY MATLAB VBPCA CODE PATH HERE
% -----------------------------
addpath("/Users/josh/Documents/VBPCA");  

% Confirm pca_full is found
if exist("pca_full", "file") ~= 2
  error("Could not find pca_full.m on the Octave path. Edit addpath(...) above.");
end

% Confirm RotateToPCA is found (you added RotateToPCA.m)
if exist("RotateToPCA", "file") ~= 2
  error("Could not find RotateToPCA.m on the Octave path. Ensure it exists and addpath(...) is correct.");
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
rand("state", 1);
randn("state", 1);

n_features = 6;
n_samples = 10;
x = randn(n_features, n_samples);
k = 3;

opts = _make_opts(200, 1, 0);

result = pca_full(x, k, opts);

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

% -------------------------------------------------------------------------
% 3) RotateToPCA fixture (captures one realistic state + rotation result)
% -------------------------------------------------------------------------
% Goal: test Python rotate_to_pca directly against legacy RotateToPCA.m.
%
% We run pca_full for a few iterations with rotate2pca=0 so we can
% capture (A,S,Av,Sv,Isv) BEFORE rotation, then call RotateToPCA once.

rand("state", 3);
randn("state", 3);

x = randn(n_features, n_samples);
k = 3;

opts = _make_opts(5, 1, 0);     % short run is fine; we just need a state
opts.rotate2pca = 0;            % CRITICAL: do NOT rotate inside pca_full

out = pca_full(x, k, opts);     % returns struct when only one output

A0 = out.A;
S0 = out.S;
Mu0 = out.Mu;
V0 = out.V;
Av0 = out.Av;
Sv0 = out.Sv;
Isv0 = out.Isv;

% obscombj isn't returned by pca_full; in uniquesv=0 mode RotateToPCA
% ignores it anyway, so pass an empty cell.
obscombj0 = {};

update_bias = 1;

[dMu, A1, Av1, S1, Sv1] = RotateToPCA(A0, Av0, S0, Sv0, Isv0, obscombj0, update_bias);

rotate_path = fullfile(outdir, "legacy_rotate_to_pca.mat");
save(rotate_path, save_format, ...
     "A0","S0","Mu0","V0","Av0","Sv0","Isv0","obscombj0","update_bias", ...
     "dMu","A1","S1","Av1","Sv1");
fprintf("Wrote %s\n", rotate_path);

fprintf("Done.\n");
