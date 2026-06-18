---
layout: post
title: Calculating Planet Volumes in Python
image: "/posts/Calculating_Planet_Volumes.jpg"
tags: [Python, Numpy, Planets]
---

### This project is about using Python and Numpy to calculate the volumes of all the planets in our solar system:
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

And there are the planets' volumes!

---

### What if we wanted the hypothetical volumes of a million random planets, though...?

---

Letting the radii hypothetically be any one million random integers, ranging from 1 to 1000, we get:

```python
radii = np.random.randint(1, 1000, 1000000)
volumes = 4/3 * np.pi * radii**3
print(volumes)
```

And just like that, we've hypothetically found the volumes of a million planets!

