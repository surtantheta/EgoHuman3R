<h1 align="center">EgoHuman3R: Bridging First-Person and Third-Person Human-Scene Reconstruction</h1>
<p align="center">
  <em>Unifying egocentric (EgoAllo) and allocentric (Human3R) human reconstruction — for holistic understanding of humans within their scenes.</em>
</p>

<p align="center">
  <a href="https://egoallo.github.io/" target="_blank">EgoAllo</a> •
  <a href="https://fanegg.github.io/Human3R/" target="_blank">Human3R</a> •
  <a href="https://projectaria.com/datasets/adt/" target="_blank">Aria Digital Twin (ADT)</a>
</p>

<hr/>



https://github.com/user-attachments/assets/54aa1324-9b52-48df-8c99-3ec0ac2dd055


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

<video autoplay loop muted playsinline controls style="width:100%;height:auto;">
  <source src="[teaser_out.mp4](https://www.youtube.com/watch?v=bdsC6Ukws6s)" type="video/mp4">
</video>

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

<h2 align="center">Rigid Alignment for Better Physics & Visualization</h2>

<p>
<b>We apply a user-controlled, single rigid transform to remove
coordinate bias introduced by sensors / SLAM.  
</p>

<ul>
  <li><b>Inference (World-Rotate)</b>: rotate the <i>entire world</i> (poses + scene) <b>before</b> sampling / optimization.</li>
  <li><b>Visualization (Scene-Rotate)</b>: rotate <i>only the scene point cloud</i> for a clean, upright view <b>after</b> inference.</li>
</ul>

<hr/>

<h4>Typical CLI Usage</h4>

```bash
   python ./scripts/3_aria_inference_fixedrot.py --traj_root ./egoallo_example_trajectories/<seq>
```
<hr/>

<h3>🪞 Visualization (Scene-Rotate)</h3>

<p>
Applies a fixed rotation to the <b>scene point cloud only</b> (e.g., +90° about X) for viewing;
poses / joints remain exactly as saved, so there is no “twist” of the human.
</p>

<h4>Typical CLI Usage</h4>

```bash
  python ./scripts/4_visualize_outputs_fixedrot.py --search_root_dir ./egoallo_example_trajectories/<seq>
```



https://github.com/user-attachments/assets/8b4629f9-0fb1-4f94-8475-d728d4b812d6


<hr/>

<h2>🧩 Integration Pipeline (High-Level)</h2>
<ol>
  <li>Extract head-pose trajectory and camera frames from ADT sequences.</li>
  <li>Run EgoAllo for initial first-person body + hand reconstruction.</li>
  <li>Run Human3R for scene and other-human reconstruction.</li>
  <li>Apply unified scale, orientation, and contact optimization to align all entities using Umeyama.</li>
</ol>

<hr/>


https://github.com/user-attachments/assets/719960c5-29c2-425f-8cf8-9ee766a2c64c

# Run EgoHuman3R

```bash
  cd scripts
  python ego_self_export.py --npz ./egoallo/egoallo_example_trajectories/Apartment_release_multiskeleton_party_seq114_71292/egoallo_outputs/<seq_id>.npz --out_dir ./egoallo/egoallo_example_trajectories/Apartment_release_multiskeleton_party_seq114_71292
  python demo_save_reload.py --model_path src/human3r.pth --size 512 --seq_path ./egoallo/egoallo_example_trajectories/Apartment_release_multiskeleton_party_seq127_M1292/Apartment_release_multiskeleton_party_seq127_M1292.mp4 --subsample 1 --use_ttt3r --reset_interval 100 --bundle_dir ./Human3R/output/Apartment_release_multiskeleton_party_seq114_71292 --export_bundle --use_ttt3r
  python egoallo_human3r.py --p1_dir ./egoallo/egoallo_example_trajectories/Apartment_release_multiskeleton_party_seq114_71292/ego_self_output --bundle_dir ./Human3R/output/Apartment_release_multiskeleton_party_seq114_71292
```

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

