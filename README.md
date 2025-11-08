<h1 align="center">EgoAllo × Human3R: Bridging First-Person and Third-Person Human-Scene Reconstruction</h1>
<p align="center">
  <em>Unifying egocentric (EgoAllo) and allocentric (Human3R) human reconstruction — for holistic understanding of humans within their scenes.</em>
</p>

<p align="center">
  <a href="https://egoallo.github.io/" target="_blank">EgoAllo</a> •
  <a href="https://fanegg.github.io/Human3R/" target="_blank">Human3R</a> •
  <a href="https://projectaria.com/datasets/adt/" target="_blank">Aria Digital Twin (ADT)</a>
</p>

<hr/>

<h2>🌐 Motivation</h2>
<p>
Recent years have seen a surge of progress in human reconstruction from egocentric and third-person viewpoints, yet these lines of work remain mostly disjoint.
</p>

<ul>
  <li>
    <strong>EgoAllo</strong> (<a href="https://egoallo.github.io/" target="_blank">project page</a>) estimates a <em>first-person</em> wearer’s full-body and hand motion from head-mounted cameras and head poses, mapping them into a world (allo-centric) frame.
  </li>
  <li>
    <strong>Human3R</strong> (<a href="https://fanegg.github.io/Human3R/" target="_blank">project page</a>) reconstructs <em>third-person</em> human meshes and scene from online stream of images, reasoning about human–scene alignment, contact, and scale.
  </li>
</ul>

<p>
Existing works such as <a href="https://sanweiliti.github.io/egohmr/egohmr.html" target="_blank">EgoHMR</a> primarily handle <em>third-person reconstruction</em> of visible humans, while EgoAllo focuses solely on <em>the camera wearer</em>.
This project proposes to <strong>marry EgoAllo and Human3R</strong> — achieving consistent reconstruction of both the <em>first-person human</em> (the camera wearer) and <em>third-person humans</em> (other people in the scene), together with the surrounding <strong>scene geometry</strong>.
</p>

<hr/>

<h2>🎯 Project Goals</h2>

<ul>
  <li><strong>Joint First-Person & Third-Person Reconstruction:</strong> Recover both the camera wearer and others in the environment under a unified coordinate frame.</li>
  <li><strong>Scene Reconstruction:</strong> Use SLAM-derived or learned point clouds to build a spatially consistent environment.</li>
  <li><strong>Cross-view Consistency:</strong> Enforce alignment between egocentric and allocentric reconstructions.</li>
  <li><strong>Temporal & Contact Coherence:</strong> Maintain smooth, physically plausible interactions between humans and the environment.</li>
</ul>

<hr/>

<h2>⚙️ Implementation Notes — EgoAllo Limitations</h2>

<p>
When implementing EgoAllo directly on the
<a href="https://projectaria.com/datasets/adt/" target="_blank">Aria Digital Twin Dataset (ADT)</a>,
we encountered several practical issues that motivated this hybrid design:
</p>

<ul>
  <li>
    <strong>Orientation Mismatch:</strong> The reconstructed human often exhibits incorrect yaw or facing direction relative to ADT’s world frame. This arises from inconsistent frame conventions between EgoAllo’s canonical frame and ADT’s coordinate system (e.g., left- vs. right-handed, different up axes, or head pose reference).
  </li>
  <li>
    <strong>Limited Third-Person Awareness:</strong> EgoAllo reconstructs only the camera wearer, with no reasoning about other humans or environmental contacts.
  </li>
</ul>

<p>
Our ongoing integration introduces <em>Human3R’s contact-aware and scale-aware optimization</em> modules
to correct these issues, aligning the reconstructed humans with both the physical scene and other agents.
</p>

<hr/>

<h2>🎥 Example: Orientation Mismatch in EgoAllo (ADT)</h2>

<p>
Below is a short video illustrating the orientation problem encountered when applying the original EgoAllo pipeline to ADT sequence, for e.g. Apartment_release_multiskeleton_party_seq106_71292. 
  
To reproduce the issue, setup EgoAllo, download the vrs file and slam for any particular sequence and run 
```bash
   python 3_aria_inference.py --traj-root ./egoallo_example_trajectories/<seq>
```

</p>




https://github.com/user-attachments/assets/0ffb6bb7-e3da-4898-8e09-60fa432f246a

<p>
Our observation is that the <strong>ground plane</strong> estimated by EgoAllo for ADT sequences is <em>not aligned</em> with the 
actual surface of the scene point cloud. Consequently, the SMPL parameter prediction becomes inconsistent with 
the true world geometry — producing a visibly <strong>“hovering” human mesh</strong> above the ground surface.
This misalignment highlights the need for consistent coordinate transformation between EgoAllo’s canonical frame 
and the ADT world frame, as well as improved use of the scene’s geometric context during optimization.
</p>

<h2>🧠 Our Proposed Tweak — Gravity-Aligned Global Frame Correction</h2>

<p>
Given the observed orientation inconsistencies and hovering SMPL meshes in EgoAllo outputs, we introduce
a <strong>gravity-aligned global rotation</strong> strategy that consistently reorients all world-space quantities
— camera trajectory, scene point cloud, and human body motion — before optimization.
This ensures the ground plane and gravity axis are correctly aligned, resulting in physically coherent reconstructions.
</p>

<hr/>

<h2 align="center">Script Type A — Rigid Alignment for Better Physics & Visualization</h2>

<p>
<b>Script Type A</b> applies a user-controlled, single rigid transform to remove
coordinate bias introduced by sensors / SLAM.  
There are two A-variants:
</p>

<ul>
  <li><b>A-Inference (World-Rotate)</b>: rotate the <i>entire world</i> (poses + scene) <b>before</b> sampling / optimization.</li>
  <li><b>A-Visualization (Scene-Rotate)</b>: rotate <i>only the scene point cloud</i> for a clean, upright view <b>after</b> inference.</li>
</ul>

<hr/>

<h3>🧩 A-Inference (World-Rotate)</h3>

<p>
Applies a fixed Euler rotation (e.g., Rx / Ry / Rz) to <b>poses and scene</b> before sampling,
with optional floor recenter.  
This improves physical priors (gravity, contact), numerical stability, and cross-sequence comparability.
</p>

<h4>Key Properties</h4>
<ul>
  <li>Rotates <code>Ts_world_cpf</code>, <code>Ts_world_device</code>, and the point cloud.</li>
  <li>Optional recenter so the rotated floor satisfies <code>z = 0</code>.</li>
  <li>Ordering convention: <code>R = Rz · Ry · Rx</code> (left-multiplied to world→* transforms).</li>
</ul>

<h4>Typical CLI Usage</h4>
<pre><code>python 3_aria_inference_fixedrot.py \
  --traj_root /path/to/run \
  --rot_x_deg 90 --rot_y_deg 0 --rot_z_deg 0 \
  --recenter_floor_zero True \
  --num_samples 1 --traj_length 4000
</code></pre>

<h4>When to Use</h4>
<ul>
  <li>Floor looks slanted or gravity axis is off in raw data.</li>
  <li>Contact / penetration / floor losses assume \(z\parallel g\) but fail in practice.</li>
  <li>You want physically meaningful coordinates before any optimization.</li>
</ul>

<hr/>

<h3>🪞 A-Visualization (Scene-Rotate)</h3>

<p>
Applies a fixed rotation to the <b>scene point cloud only</b> (e.g., +90° about X) for viewing;
poses / joints remain exactly as saved, so there is no “twist” of the human.
</p>

<h4>Key Properties</h4>
<ul>
  <li>Rotates only the scene: <code>P' = P · R_x(θ)^T</code> (row-vector convention).</li>
  <li>Optional recenter scene so floor at <code>z = 0</code>.</li>
  <li>Ideal for upright visualization without altering results.</li>
</ul>

<h4>Typical CLI Usage</h4>
<pre><code>python 4_visualize_outputs_scene_rx90.py \
  --search_root_dir /path/to/run \
  --rot_x_deg 90 \
  --recenter_floor_zero True
</code></pre>

<h4>When to Use</h4>
<ul>
  <li>Outputs are correct but appear tilted in the viewer.</li>
  <li>You need a cosmetic, non-destructive re-orientation for demos or debugging.</li>
</ul>

<hr/>

<h3>🧭 Practical Notes</h3>
<ul>
  <li><b>Rotation order matters</b>: use a consistent Euler sequence (e.g., <code>Rz·Ry·Rx</code>).</li>
  <li><b>Row-vector vs column-vector</b> conventions differ; these scripts use row-vector rotations for point clouds (<code>P' = P·R^T</code>).</li>
  <li><b>Recenter strategy</b>: set floor to <code>z = 0</code> using a robust percentile (e.g., median z).</li>
</ul>

<hr/>

<h3>📘 TL;DR</h3>
<ul>
  <li><b>A-Inference</b> = rotate world (poses + scene) <i>before</i> optimization → physics-aligned coordinates.</li>
  <li><b>A-Visualization</b> = rotate scene <i>only</i> <i>after</i> inference → upright, interpretable viewing.</li>
</ul>

<blockquote>
<b>Unified Principle:</b><br/>
Apply a single rigid alignment to eliminate arbitrary coordinate bias and make gravity, floor, and contacts coherent with the chosen axes.
</blockquote>


<!-- Alternative: YouTube thumbnail link
<p>
  <a href="https://www.youtube.com/watch?v=YOUR_VIDEO_ID" target="_blank">
    <img src="https://img.youtube.com/vi/YOUR_VIDEO_ID/hqdefault.jpg" alt="EgoAllo ADT Orientation Demo" width="720"/>
  </a>
</p>
-->

<hr/>

<h2>🧩 Integration Pipeline (High-Level)</h2>
<ol>
  <li>Extract head-pose trajectory and camera frames from ADT sequences.</li>
  <li>Run EgoAllo for initial first-person body + hand reconstruction.</li>
  <li>Run Human3R for scene and other-human reconstruction.</li>
  <li>Apply unified scale, orientation, and contact optimization to align all entities.</li>
  <li>Visualize results and compare against ADT ground truth.</li>
</ol>

<hr/>

<h2>🚀 Quick Start</h2>
<pre><code># environment
conda create -n egoallo_human3r python=3.10 -y
conda activate egoallo_human3r
pip install -r requirements.txt

# run example on ADT
python run_hybrid_pipeline.py \
  --adt_seq /path/to/ADT/sequence \
  --out_dir outputs/demo_sequence \
  --viz
</code></pre>

<hr/>

<h2>🧭 Roadmap</h2>
<ul>
  <li>Finalize ADT↔EgoAllo coordinate alignment and yaw correction.</li>
  <li>Integrate Human3R’s contact and scale priors for scene-aware optimization.</li>
  <li>Evaluate across multiple ADT sequences with quantitative metrics (MPJPE, PVE, orientation error).</li>
  <li>Release visual demos and detailed ablations.</li>
</ul>

<hr/>

<h2>📚 References</h2>
<ul>
  <li><a href="https://egoallo.github.io/" target="_blank">EgoAllo: Egocentric-to-Allocentric Human Motion Reconstruction</a></li>
  <li><a href="https://fanegg.github.io/Human3R/" target="_blank">Human3R: Human Reconstruction with Human–Scene Reasoning</a></li>
  <li><a href="https://sanweiliti.github.io/egohmr/egohmr.html" target="_blank">EgoHMR: Egocentric Human Mesh Recovery</a></li>
  <li><a href="https://projectaria.com/datasets/adt/" target="_blank">Aria Digital Twin Dataset</a></li>
</ul>

<hr/>

<h2>🤝 Acknowledgements</h2>
<p>
We build upon the open-source releases of
<a href="https://egoallo.github.io/" target="_blank">EgoAllo</a> and
<a href="https://fanegg.github.io/Human3R/" target="_blank">Human3R</a>.
We thank the Meta Project Aria team for the <a href="https://projectaria.com/datasets/adt/" target="_blank">ADT dataset</a>.
</p>

<hr/>

<h2>📄 License</h2>
<p>
Specify your license here (e.g., MIT). See <code>LICENSE</code> for details.
</p>

