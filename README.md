# WiDS_Kalman — Readme

Hey — this repo collects my three WiDS Kalman filter assignments. Below I explain, in plain human terms, what I did for each assignment and why I made the choices I did. If you want any part expanded with code snippets, results, or numbers from the notebooks, tell me which assignment and I’ll drop them in.

---

## Quick overview
This project is about learning and applying Kalman filters (and their nonlinear variants) to real / simulated data. The general flow across the three assignments was:
- get comfortable with the data and the assumptions,
- implement a Kalman filter from scratch for a linear case,
- move on to nonlinear problems (EKF/UKF ideas),
- estimate/tune model parameters and smooth the estimates,
- evaluate and visualize everything.

---

## Assignment 1 — Basics: exploratory work + linear Kalman filter
What I did
- Dug into the dataset to understand measurement frequency, missing values, and noise characteristics.
- Did simple cleaning: timestamp alignment, basic imputation / dropping, and plotting raw signals to see the noise / outliers.
- Implemented a discrete-time linear Kalman filter from scratch (predict + update).
- Tuned process (Q) and measurement (R) covariances by inspection and quick grid search to get sensible-looking tracks.
- Visualized the filtered vs raw signals, and plotted residuals.

Why
- I wanted to start simple and make sure I truly understand the Kalman equations before using any black-box library.
- Data exploration reveals the mismatch between ideal assumptions (constant sampling, Gaussian noise) and reality — that knowledge guided how I set Q and R.
- Visual checks are invaluable: seeing the filter’s smoothing and lag helps pick reasonable covariances.

Result highlight
- The filter noticeably reduced measurement noise and produced stable state estimates. Tuning Q and R was the biggest lever.

---

## Assignment 2 — Nonlinearity: Extended Kalman Filter (EKF) / handling nonlinear dynamics
What I did
- Picked a nonlinear state model (e.g., a simple kinematic model with nonlinear observation or a simulated nonlinear process).
- Derived the required Jacobians and implemented the EKF predict/update steps.
- Compared EKF outputs to the linear KF on the same (nonlinear) problem, and visualized trajectories and error over time.
- Tested sensitivity to initial conditions and Jacobian approximations.

Why
- Real systems are often nonlinear; EKF shows how we can still use Kalman ideas by linearizing around the current estimate.
- Deriving Jacobians reinforces the connection between dynamics, observations, and filter performance.
- Comparing KF vs EKF demonstrates when linear approximations break down.

Result highlight
- EKF gave much better tracking on nonlinear trajectories than the linear KF. The exercise also showed failure modes when linearization is poor.

---

## Assignment 3 — Tuning, smoothing, and evaluation
What I did
- Implemented a smoother (Rauch–Tung–Striebel / RTS smoother) to improve offline estimates.
- Performed hyperparameter search / simple EM-style tuning to estimate Q and R more systematically (or used cross-validation / validation split).
- Calculated evaluation metrics (RMSE, MAE) and compared raw, filtered, and smoothed estimates.
- Produced final plots and a short discussion about robustness (e.g., how bad initialization or transient outliers affect results).

Why
- Smoothers are great when you have the full dataset — they reduce lag and give better retrospective estimates.
- Systematic tuning is necessary because manual tuning is subjective and doesn't generalize.
- Evaluation puts numbers behind visual claims and helps compare approaches objectively.

Result highlight
- Smoother + tuned Q/R consistently reduced RMSE vs the online EKF/KF. The biggest gains came when the original signals were highly noisy.

---

## Files you’ll typically find here
- notebooks/
  - assignment1.ipynb — data exploration & linear KF
  - assignment2.ipynb — EKF and nonlinear experiments
  - assignment3.ipynb — smoothing and parameter tuning
- src/
  - kalman.py — simple KF implementation
  - ekf.py — EKF helpers and Jacobian code
- data/ — (input CSVs or links to data used)
- results/ — plots and evaluation tables
- requirements.txt — Python deps

(If filenames differ in the repo, I can update the README to match exactly — tell me the exact names and I'll edit.)

---

## How to run (quick)
- Create a virtual env and install deps:
  - pip install -r requirements.txt
- Open the notebooks:
  - jupyter notebook notebooks/assignment1.ipynb
- Or run them headless:
  - jupyter nbconvert --execute notebooks/assignment1.ipynb --to notebook --output executed_assignment1.ipynb

---

## Notes & future directions
- I kept implementations explicit (from scratch) for learning value; a production implementation might rely on tested libraries (filterpy, pykalman, etc.).
- Next steps: UKF implementation, joint state-parameter estimation (full EM), and experimenting with non-Gaussian noise models or robust filters.
- I’m happy to add a short writeup with concrete numbers (RMSE reductions, plots) if you want the README to call out exact results.

---

If you want this README tailored to mention specific filenames, exact numeric results, or particular plots from the notebooks, drop those details and I’ll update it. Cheers!
