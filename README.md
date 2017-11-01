# Code

### Behavior Simulator
- Parameters can be adjusted in the init-function
- To generate new behavior, do:
  bg = BehaviorGenerator()
  bg.sample(n=10000, size=5000000)
  with open('<where to save the file>', 'wb') as f:
      pickle.dump([bg.data, bg.duration, bg.gt, bg.etype], f)
