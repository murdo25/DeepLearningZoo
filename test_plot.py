print('plot test')
import time
from matplotlib.pyplot import plot, ion, show
ion() # enables interactive mode
plot([1,2,3]) # result shows immediatelly (implicit draw())

for i in range(100):
    time.sleep(1)

# at the end call show to ensure window won't close.
show()










Use matplotlib's calls that won't block:

Using draw():

from matplotlib.pyplot import plot, draw, show
plot([1,2,3])
draw()
print 'continue computation'

# at the end call show to ensure window won't close.
show()
Using interactive mode:

from matplotlib.pyplot import plot, ion, show
ion() # enables interactive mode
plot([1,2,3]) # result shows immediatelly (implicit draw())

print 'continue computation'

# at the end call show to ensure window won't close.
show()




lst = []
ion()

show()

# --- place your optimization loop here
for i in range(1000):
    print(i)
    output = [i]
    
    lst.append(output[0])
    
    plot(lst)

