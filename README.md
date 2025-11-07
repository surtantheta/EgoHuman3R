<!--
  README (HTML-flavored) — paste directly into README.md
  Notes:
    • GitHub Markdown supports inline HTML.
    • For videos: either store an .mp4 in the repo and use <video>, or link to YouTube/Vimeo with a thumbnail.
-->

<h1 align="center">EgoAllo → Aria Digital Twin (ADT) Integration</h1>
<p align="center">
  <em>Estimating full-body & hand motion from egocentric input, adapted to Project Aria’s Digital Twin dataset.</em>
</p>

<p align="center">
  <a href="https://egoallo.github.io/" target="_blank">EgoAllo Project Page</a> •
  <a href="https://projectaria.com/datasets/adt/" target="_blank">Aria Digital Twin (ADT)</a>
</p>

<hr/>

<h2 id="overview">Overview</h2>
<p>
  This repository applies <a href="https://egoallo.github.io/" target="_blank">EgoAllo</a> to the
  <a href="https://projectaria.com/datasets/adt/" target="_blank">Aria Digital Twin (ADT)</a> dataset.
  EgoAllo estimates the wearer’s body and hands from egocentric imagery and head motion, producing
  a globally consistent skeleton in the scene (allo-centric) frame. Here we evaluate and adapt EgoAllo’s
  pipeline for ADT’s coordinate conventions and device streams.
</p>

<h3 id="goal">Main Goal</h3>
<ul>
  <li>Assess generalization of EgoAllo to ADT and ensure outputs (skeletons, hands) are correctly oriented, scaled, and placed in ADT’s world frame.</li>
</ul>

<h3 id="inputs">Inputs</h3>
<ul>
  <li>Egocentric camera frames (from Aria device)</li>
  <li>Head trajectory (6-DoF) / SLAM poses</li>
  <li>Hand keypoint detections (e.g., wrist/palm/hand landmarks)</li>
  <li><em>(Optional)</em> Calibration and floor/world references provided by ADT</li>
</ul>

<h3 id="outputs">Outputs</h3>
<ul>
  <li>Full-body skeleton (incl. hands) in the ADT world coordinate frame</li>
  <li>Per-frame poses suitable for visualization and quantitative comparison to ADT mocap</li>
  <li><em>(Optional)</em> Height estimate and diagnostic overlays</li>
</ul>

<hr/>

<h2 id="adt-adaptation">ADT Adaptation</h2>
<p>
  We interface the Aria device streams (images + head pose) with EgoAllo’s input format,
  then export all estimated poses into the ADT world frame for qualitative and quantitative checks
  against the dataset’s ground-truth motion capture.
</p>

<h3 id="orientation-issue">Orientation Issue We Observed</h3>
<p>
  When running the unmodified EgoAllo code on ADT sequences, we noticed a <strong>systematic orientation mismatch</strong>:
  the estimated body location/scale look reasonable, but the <strong>facing direction (yaw)</strong> can be misaligned
  relative to ADT’s world frame / mocap.
</p>
<ul>
  <li><strong>Symptoms:</strong> Body appears rotated or facing the wrong direction despite correct placement.</li>
  <li><strong>Likely cause:</strong> Frame convention differences between EgoAllo’s canonical frame and ADT’s world frame
      (e.g., forward axis / up axis, CPF/head reference, left-handed vs. right-handed systems).</li>
  <li><strong>Status:</strong> We are adding a consistent frame remapping step and validating across multiple sequences.</li>
</ul>

<!-- ====== VIDEO SECTION ====== -->
<h2 id="demo-video">Demo Video</h2>

<!-- Option A: Repo-hosted MP4 (recommended if you store the .mp4 in the repo) -->
<!-- Replace path with your actual file path, e.g., docs/adt_orientation_demo.mp4 -->
<video controls width="720" muted>
  <source src="PATH/TO/YOUR/VIDEO.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>

<!-- Option B: External video (YouTube/Vimeo) with a clickable thumbnail -->
<!--
<p>
  <a href="https://www.youtube.com/watch?v=YOUR_VIDEO_ID" target="_blank">
    <img src="https://img.youtube.com/vi/YOUR_VIDEO_ID/hqdefault.jpg" alt="ADT Orientation Mismatch Demo" width="720"/>
  </a>
</p>
-->

<hr/>

<h2 id="quickstart">Quick Start</h2>
<ol>
  <li>Clone this repository.</li>
  <li>Download and prepare an ADT sequence (images, 6-DoF head trajectory, calibration).</li>
  <li>Install dependencies (match EgoAllo requirements + ADT parsing tools).</li>
  <li>Configure the I/O paths in the provided scripts.</li>
  <li>Run inference and export world-frame poses for visualization/evaluation.</li>
</ol>

<pre><code># example (placeholder)
conda create -n egoallo_adt python=3.10 -y
conda activate egoallo_adt
pip install -r requirements.txt

# run
python run_egoallo_adt.py \
  --adt_seq /path/to/ADT/sequence \
  --out_dir outputs/sequence_xyz \
  --viz
</code></pre>

<hr/>

<h2 id="roadmap">Roadmap</h2>
<ul>
  <li>Finalize a robust ADT↔EgoAllo frame remapping (consistent up/forward, yaw alignment).</li>
  <li>Quantify orientation and joint errors against ADT mocap; add scripts and tables.</li>
  <li>Improve hand pose alignment in world frame; optional scale calibration checks.</li>
  <li>Release minimal repro scripts for others to validate on ADT.</li>
</ul>

<hr/>

<h2 id="citation">Citation</h2>
<p>Please cite the original EgoAllo work if you use this integration in research.</p>

<pre><code>@inproceedings{EgoAllo2024,
  title     = {EgoAllo: ...},
  author    = {...},
  booktitle = {...},
  year      = {2024}
}
</code></pre>

<hr/>

<h2 id="acknowledgements">Acknowledgements</h2>
<p>
  Built on <a href="https://egoallo.github.io/" target="_blank">EgoAllo</a>.
  Thanks to the <a href="https://projectaria.com/datasets/adt/" target="_blank">Project Aria</a> team for ADT.
</p>

<hr/>

<h2 id="license">License</h2>
<p>
  Specify your license here (e.g., MIT). See <code>LICENSE</code> for details.
</p>

