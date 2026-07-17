

[] use for guiding
  [] train scanpath-conditioned model so the image will generate with relevant portions
      at your latest fixations.
        [] morph generations and/or only update on long saccades
  [] train point-conditioned video model (longcat? ltx?) so that gaze naturally moves media
[] use for evading gaze
  [] could cache generated images/video by coordinate
  [] detect saccades?
[] use for optimizing contents
  [] using those simple shape morphers
  [] using zoom-in and propogate
  [] using preference-prior-esque genrec system


[x] salience maps dataloading (https://arxiv.org/abs/1505.03581)
[x] klein training
    [x] ensure we can load klein & lora/finetune/etc. *period*.
    [x] remove text experts
    - could enforce a maximum number of points on the scanpath?
      [x] it's just RoPE values, so not a compute optimization.
    [x] patch pipe and setup qual eval
    [x] ckpt loading (see other notimplementederror)
    [x] condition on scanpath (not just fixation points)
    - if you don't keep-distill train, would need to set a timestep schedule


[x] klein keep-distill training (https://arxiv.org/abs/2605.05204)
  [x] cache teacher outputs
    - vae latents saved as tensor with mapping from images
  - may be overfitting worse by only single trajectories per image
    [] could use LoRA & switch on/off for student / teacher
  [x] qlora, train with adam, etc.
  [] support batch_size > 1
  [x] option to add back text encoder
    [x] can fall back to diffusers klein+pipe & patch the RoPE
  [x] use existing ckpt for guiding (boxes below) to proof out further changes.
  - put onto vast.ai for full finetune?
[x] inference with saccade updating our image

Next:
[] train at low-ish-res but at AR that matches monitor
  [] regenerate teacher i/o pairs
[] light cfg on with/without gaze rope?
