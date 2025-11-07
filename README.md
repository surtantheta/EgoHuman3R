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

<h2>📦 Inputs & Outputs</h2>

<h3>Inputs</h3>
<ul>
  <li>Egocentric RGB streams from Aria or similar HMD devices.</li>
  <li>6-DoF head-pose trajectory from device SLAM.</li>
  <li>Hand keypoint detections from egocentric views.</li>
  <li>Third-person scene observations (RGB, depth, or reconstruction meshes).</li>
</ul>

<h3>Outputs</h3>
<ul>
  <li>Full-body mesh of the <em>first-person human</em> (camera wearer) in world coordinates.</li>
  <li>Meshes of <em>third-person humans</em> reconstructed and aligned with the same world frame.</li>
  <li>Reconstructed scene geometry and contact-aware alignment between all entities.</li>
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
  <li>
    <strong>Scene Ignorance:</strong> EgoAllo does not incorporate explicit 3D scene geometry during optimization, leading to occasional floating or mis-scaled reconstructions.
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
To reproduce the issue, setup EgoAllo, download the vrs file and slam for any particular sequence.
</p>
<video src="./data/sample.mp4" width="320" height="240" controls></video>

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

