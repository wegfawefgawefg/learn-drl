ARTICLE FOR PEOPLE WHO HAVE GOTTEN STUCK ON IMPLEMENTATIONS OF EXPERIENCE REPLAY

Dont Reach Too Far Too Early: 
    So Let's Make The Minimum Viable Experience Replay Buffer And Ruin It With Spice

I see a lot of drl agent implementations online are using experience replay buffers made out of 
a bunch of numpy arrays. (struct of arrays (link)) (example) It is actually a pretty good 
solution if your batch sizes are big. So you should definitely just copy 
and paste it into your code blindly and then struggle with the shape errors for like 3 hours. (a valid way to learn)

Anyways, an important thing to remember is that using numpy to store your agent's memories is just an optimization.
It is absolutely not necessary if you are new to reinforcement learning.
To use an optimization is almost always to sacrifice program simplicity for performance. 
When it comes to optimizations, you are almost always better off proving the thing works with the 
unoptimized version first, because every line you don't have is a line that can't break.

There is no shame in finding the fancy vectorized numpy memories to be complicated and confusing.
They are. 
I few times now I have spent 30 minutes to a few hours trying to write in numpy what i could have done 
in a python for loop in less than 5 minutes. Why is all the indexing so complicated? What are all these dtype things? 
Why do you have to index pytorch tensors with int64's instead of int32's? Who knows? I sure don't.
Did you give up on life after seeing phil's learn function for the dueling double deep q network? Me too.
I have a small head. Therefore my brain is small. I can basically only work with 200 lines max before i start urinating myself.

So, in this tutorial we are gonna build an experience replay buffer so dumb, even I can understand it.
This is the minimum possible reinforcement learning memory that you should be using in your very first deep q network or actor-critic agent, 
or atleast up until you see any serious performance bottlenecks caused by it.

memory = []

And thats the end of the tutorial. Thanks for reading.





Seriously, though that’s the "Experience Replay Buffer" the alleged """scientists""" keep trying to tell us about.
"How do i add a new memory to it?" you may ask.

newMemory = (state, action, reward, nextState)
memory.append(newMemory)

The memories are often called "transitions", because they are transitions in time.
If a single memory consists of a chain of transitions that are contiguous in time, it is called a "trajectory"

"Wait... arent you going to be duplicating a lot of data, wasting my ram?"

"How do i get 50 random memories from the memory?"

import random
randomMemories = random.choices(memory, k=50)

"How do i fetch the reward from a specific memory?"
reward = randomMemory[2]

"Thats... kind of ugly. Someone looking at that won't know what 2 means. Can I make that maybe a little less dumb?"
Yes, using classes or namedtuples, but remember earlier what i said about optimizations.
If you give someone a file longer than 200 lines, don't expect them to read it unless you are paying them.
There is no moore's law for the human attention span. 

"fine, you tactless brute. how do i fetch all the rewards in an array"
memories = np.stack(randomMemories)
rewards = memory[:, 2]  #   numpy indexing magic.     equivalent to:
                            #   rewards = []
                            #   for i in len(memory):
                            #           rewards.append(memory[i][2])

"Can you give me an example of using this memory in a learn function?"
def learn():    #   incredibly clausterphobic learn function
    randomMemories = random.choices(memory, k=50)       #   fetch random memories
    memories = np.stack(randomMemories)                 #   stack in numpy array
    states, actions, rewards, nextStates = memories.T   #   extract into seperate arrays
    qvals, qvals_ = net.forward(states), net.forward(nextStates)    #   please dont code like this
    td = rewards + np.amax(qVals_, axis=1)              #   numpy magic equivalent to:
    loss = genericLossFunction(td, qvals[actions])          #   td = []
    loss.backward()                                         #   for i in len(memories):
    loss.step()                                             #       td.append( rewards[i] + max(qVals_[i ]))
    #   one hop this time

and there you go.
a fully functional experience replay buffer in one line.

Now here are some ways to spice it up.
But as you go through the naughty next section remember what the great philosopher Confucius once said:
"""If you ask someone to google 5 lines of code they might google 5 lines of code after an hour of netflix. 
If you ask someone to google 10 lines of code, you might as well have asked them to translate the bible to chinese."""
                                                        -Albert Confucius, 492 BC

Spice 1:
    Named Tuples

If you hate indexing the transition values by number ( reward = randomMemory[2] ) you can use Named Tuples 
which are just like regular tuples except they lets you access things by name.

from collections import namedtuple

#	define your named-tuple
SARS = namedtuple()
#   use your newly made named-tuple type to make a memory
aMemory = SARS(state=(10.3, 0.4, 0.2, 0.5), 
               action=2, 
               reward=10.0, 
               nextState=(10.4, 0.5, 0.2, 0.4))

memory.append(aMemory)
#   use it like this
veryFirstMemory = memory[0]
state = veryFirstMemory.state       #   a maternity ward
reward = veryFirstMemory.reward     #   life
action = veryFirstMemory.action     #   cry and scream

Downsides:
    Named tuples are implemented underneath with a python dictionary. So when you access them you aren't just accessing a 
    tuple like an array, you are doing a hash key lookup ( like dict["key"]). So its slower if you abuse it.

    A lot of people don't know about named tuples in python. If you show your friend SARS() they are gonna think you made 
    a SARS class somewhere. Actually you could just make a memory class, but putting random container classes everywhere is 
    more lines. And if my friend sends me a 1000 line C++ file on discord, I'm not going to read it.
    Anyways if you just listened to me in the first place and wrote less code we wouldn't be arguing about this.

Spice 2:
    Show Her How Big Your Deque Is

Usually you want your memory to have a max size so you dont fill up your ram completely. (Note to self we should do an article 
on the effects of minimum and maximum buffer size). 
The other day I saw someone manually draining their memory array like this:

overBudget = len(memory) - MAX_MEMORY_SIZE
memory = memory[overBudget:]

That works but its really dank. An alternative would be to replace your memory list with a memory deque:

#   instead of:
memory = []

#   do
from collections import deque
memory = deque(maxlen=100)

The deque will automatically drain the oldest entries as you keep adding new items.

Downsides:
    There is really no downside actually. Deques are great. 1 line change. No catch.

Spice 3:
    Unduplicating The States In The Transitions

As you probably noticed, the state is the largest part of a memory, compared to reward and action which are usually just one number.
Since you are so smart you probably also noticed we are storing that biggest part twice and wasting half our ram.

memory1 = (state_t0, action, reward, state_t1)  #   t1 here
memory2 = (state_t1, action, reward, state_t2)  #   t1 again here,      t2 here
memory3 = (state_t2, action, reward, state_t3)  #   t2 again here...
memory.append(memory1)
memory.append(memory2)
memory.append(memory3)

Yes. Each state other than the first and last end up stored twice, which might seem more than mildly stupid. 
Even the hoity toity numpy memories do this. Can you fix that? Yeah, probably.
    
During learn() you pick your memories for your batch at random. So instead of just grabbing some random memories, 
you would have to pick random indecies instead. Then add one to those indecies and fetch the corresponding next state.
Not impossible.

Downsides:
    There are a lot of downsides.
    The code will now be much more fickle. It will be easy to mess up the next_state fetching, and the memory would need 
    to be one larger than needed to hold the next. You wont know what the action or reward is until after you step 
    the environment, so if an episode ends you will have to store partial transitions (state, None, None, None)... 
    Basically room for bugs, that nobody asked for.

    Also PPO, TD3, A2C, A3C all commonly use either multiple worker agents or environments.
    It will be even more annoying to mix their memories togethor if the next_state's are detached.

    I haven't seen anyone do this yet.

In conclusion, I hope i helped you to do it the baby way first. Convert it to the numpy way later.
Your code should be tiny. Like 100 to 200 lines tiny.
If your code is small you can actually hold every line in your mind at once. 
Then you can focus on understanding the entire program holistically, top to bottom.