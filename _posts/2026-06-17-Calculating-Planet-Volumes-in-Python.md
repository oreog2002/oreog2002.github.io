---
layout: post
title: Calculating Planet Volumes in Python
image: "/posts/coffee_python.jpg"
tags: [Python, Numpy, Planets]
---

#### This project is about using Python and Numpy to calculate the volumes of all the planets in our solar system:
---

Add the library, Numpy, since that's going to be our biggest tool for calculating the planets' volumes.

```python
import numpy as np
```

Create a variable that'll contain the radii of all the solar system's planets from Mercury to Neptune.

```python
radii = np.array([2439.7, 6051.8, 6371, 3389.7, 69911, 58232, 25362, 24622]) 
```

Now, we'll calculate the volumes of the planets, with "np.pi" as our pi and our "radii" variable as our radii. Then, we'll show the resulting volumes using the "print" command.

```python
volumes = 4/3 * np.pi * radii**3
print(volumes)
```



---

And there are the planets' volumes!
