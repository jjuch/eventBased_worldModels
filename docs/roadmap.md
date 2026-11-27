# Roadmap: SE(3) latent world model

Living document. It records the design principles, the planned steps with acceptance criteria,
and ideas we considered and rejected. It is not a to-do list for code assistants: implement a step
only when asked.

## 1. Where we are

Stage 1 with a fixed camera showed that an observer trained on rendered RGB frames recovers the
SE(3) state of a marked ball: position 2.7 mm, orientation 0.6°, linear velocity R² ≈ 0.99,
angular velocity R² ≈ 0.99, linear acceleration R² 0.83–0.94, angular acceleration R² ≈ 0.92.
Learned acceleration beats a gravity-only rollout (16.9 vs 26.8 mm and 11.9° vs 34.6° after
140 ms). The kinematic-consistency loss acts as a regulariser.

Evaluating on a cube (`cube_tetr4`) exposed the limits of that formulation:

- Position failed by a 0.6 m offset along the viewing ray (9 mm sideways): the observer infers
  depth from apparent size, calibrated on one object. Monocular size and depth are ambiguous.
- Orientation (16° median) and angular velocity (R² 0.6–0.8) degraded although the markers were
  identical: the network used sphere-specific appearance, not only the markers.
- Targets were absolute world coordinates from one fixed camera, so the camera pose was memorised.
- The motion heads were not tied to the predicted poses and memorised per-trajectory constants.

The perception result stands. The formulation of the state was too narrow.

## 2. Design principles

1. **Kinematics is hard-coded or enforced, dynamics is learned.** How a pose evolves under a twist,
   and how frames compose, holds for every rigid body and through contacts. Forces (gravity, air,
   contact) and object properties are learned. No force law in the network or the losses.
2. **Relative to what, expressed in which coordinates.** Motion is measured relative to the static
   scene (inertial, where physics is simple) and expressed in observer-centred coordinates
   (observable, no world origin). Gravity is a property of the static scene, expressed in
   observer coordinates.
3. **No arbitrary choices in the targets.** No world origin, no starting camera pose. Each window
   uses its own anchor frame: the camera frame at the last context frame, frozen.
4. **Every body is a rigid body.** The static scene, the observer (camera) and each object are
   SE(3) bodies; they differ only in their dynamics prior: stationary, action-driven, or
   free flight plus contact. Ego-motion is the motion of the static scene seen from the camera,
   inverted.
5. **A body is an SE(3) state plus body-frame features.** Pose, twist and acceleration are what
   explain how a body's features (corners, edges, prints, or markers on a symmetric body) move in
   the image: a feature's image position is its body-frame position projected through
   camera ∘ ego ∘ object poses. The static scene's features drive ego-motion. The same body-frame
   features later describe contact geometry for the GNN. Markers are only a special case.
6. **Observable now, or a latent revealed by motion.** Appearance gives a prior; dynamics update
   it. Object size (hence depth scale), mass ratios, restitution, friction and the spin of a
   featureless sphere are latents with uncertainty, sharpened over time. Gravity fixes an object's
   metric scale through its free-fall curvature; contact links an object's scale to the scene's.
   Scale is per body: the static scene and ego-motion share one, each moving object has its own.
7. **Symmetry is not a problem.** For a symmetric body only orientation modulo its symmetry group
   is observable, and its dynamics have the same symmetry. Evaluate with symmetry-aware errors.
8. **Own motion is the action.** In an action-conditioned world model, the observer's future motion
   is given (planned), not predicted by the physics. Object dynamics never need the camera motion.
9. **Structured part, free part.** Each body slot = SE(3) state (integrated by the dynamics) +
   free embedding (appearance, shape, features, uncertainty, physical latent `c`). The SE(3) state
   changes only through the dynamics network, which may read `c`, so the free part cannot quietly
   do the dynamics.

## 3. Three teaching signals

| signal | examples | where it works |
|---|---|---|
| privileged ground truth | anchor-frame poses, twists, accelerations of camera and object; gravity; wall planes | simulation only (scaffold) |
| structural identities | pose–twist and twist–acceleration consistency; pose composition; static points stay fixed in the anchor frame; gravity constant in the anchor frame; feature projection | everywhere, no labels |
| future-observation prediction | future frames or future features | everywhere, no labels |

Train with all three in simulation; keep the last two on real video.
Dynamics can be trained on simulator states without images, and perception trained to produce
the state the dynamics consume (privileged-teacher pattern). This lets contact work start before
perception is finished.

## 4. Steps

Each step changes one thing and has an acceptance test. Earlier datasets stay as regression tests.

### Step 0: frame change only (fixed camera)
Anchor-frame targets on `ball_magnus`. Accept: metrics match the world-frame runs.

### Step 1: anchor-frame observer with a moving camera (current)
Two bodies, camera and object, both in the anchor frame; gravity in the anchor frame; soft
pose–twist and composition losses. Data: smooth camera motion over a textured static scene,
including the hard cases (camera tracking the ball; camera moving with the ball) and a static
subset. Accept:
- fixed-camera subset matches step 0;
- object metrics do not degrade with camera motion (ego/object motion separated);
- gravity direction error reported, and rollouts in the anchor frame beat gravity-only.

### Step 2: scale as an explicit quantity
Scale-free translation (bearing + inverse depth, or units of object size) plus a metric scale per
body. Shape and size randomisation. Accept: on objects of unseen size, bearing and orientation hold
and the scale error shrinks over the window (gravity as a ruler).

### Step 3: features instead of markers
Objects with intrinsic asymmetry (printed faces, seams, unequal boxes) and plain symmetric objects.
Lift the image-space soft keypoints to body-frame features; add the feature-projection identity.
Accept: markerless objects reach marker-level accuracy where observable; symmetric objects are
scored modulo symmetry.

### Step 4: belief state
Recurrent observer with uncertainty (mean and covariance in the SE(3) tangent space, plus `c`),
updated every frame. Accept: uncertainty is calibrated and shrinks as motion is observed.

### Step 5: contact
Wall planes from the static scene; contact GNN in the anchor frame over body-frame features;
restitution, friction and unseen spin as latents inferred from the bounce. Train the GNN on
simulator states first. Accept: post-contact rollouts beat a generic latent world model.

### Step 6: several objects and self-supervision
Object slots, object–object contact, future-observation losses, first real videos.

## 5. Considered and rejected

- **World-frame targets with a moving camera:** world origin is unobservable; forces memorisation.
- **Object relative to camera only:** a moving camera adds fictitious forces to the dynamics.
- **One flat latent vector:** fixed object count, no shared dynamics, identity heads.
- **Fully unsupervised latent, hoping SE(3) emerges:** removes the research claim; auxiliary only.
- **Dense scene reconstruction in the latent:** physics needs a few contact surfaces, not depth maps.
- **Relative accelerations as targets:** mix Coriolis and centrifugal terms of the camera into the
  object's dynamics.
- **A hand-built PnP or image-processing front end:** perception stays learned; structure lives in
  the state and the losses.
- **Force laws in the model (Magnus or drag coefficients as heads):** would fit this simulator,
  teach nothing general. At most an interpretability probe.
- **A constant-acceleration fit as the motion estimator:** wrong across contacts; evaluation
  diagnostic only.

## 6. Open questions

- How smooth must camera motion be for ego-motion to stay easy at 100 fps?
- Does the camera head need its own pathway (global scene features) separate from the object read-out?
- Uncertainty representation: Gaussian in the tangent space, or samples?
- What does the dynamics model consume: the SE(3) state only, or also the features?
