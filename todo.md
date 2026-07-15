

[] use for evading gaze
  [] could cache generated images/video by coordinate
  [] detect saccades?
[] use for optimizing contents
  [] using those simple shape morphers
  [] using zoom-in and propogate
  [] using preference-prior-esque genrec system
[] use for guiding
  [] train point-conditioned video model so that gaze naturally moves media
  [] train scanpath-conditioned media so the image will generate with relevant portions
      at your latest fixations.
        [] morph generations and/or only update on long saccades


[x] salience maps dataloading (https://arxiv.org/abs/1505.03581)
[] klein training
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
  [] put onto vast.ai so we can easily finetune with text encoder kept & adam
  [] could use LoRA & switch on/off




