

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



[x] salience maps dataloading (https://arxiv.org/abs/2605.05204)
[] klein keep-distill training (https://arxiv.org/abs/2605.05204)
    [x] ensure we can load klein & lora/finetune/etc. *period*.
    [x] remove text experts
    [] cache teacher outputs
    - could enforce a maximum number of points on the scanpath?
      [x] it's just RoPE values, so not a compute optimization.
    [x] patch pipe and setup qual eval
    [x] ckpt loading (see other notimplementederror)
    [x] condition on scanpath (not just fixation points)


