## My notes on some of the ReAct paper

### Agent interacting with the environment
1. At time step *t* the agent receives an observation
			$o_t \, \epsilon \,  O$
2. The agent assembles a history (or context)
			$c_t = \bigl(o_1,\,a_1,\;o_2,\,a_2,\;\dots,\;o_{t-1},\,a_{t-1},\;o_t\bigr),$
   which contains all past observations and actions up to and including the current observation.
3. An action $a_t$ is selected, using the agent's policy $\pi$ conditioned on the context $c_t$
			$\pi\bigl(a_t \mid c_t\bigr),$
	Learning a policy mapping $context \rightarrow action$ is compute intensive when that mapping must implicitly encode complex reasoning over a long trajectory context.

### ReAct
* The idea of ReAct is to augment the agent's action space  $Aˆ = A ∪ L$, where $L$ is the space of language. An action $â_t ∈ L$ in the language space, referred to as a *thought* or *reasoning trace*, does not affect the external environment, thus leading to no observation feedback. A thought $â_t$ composes useful information by reasoning over the current context $c_t$ and updating the context with the thought $â_t$ to support future action or reasoning. 
* By interleaving thought steps into the decision process, ReAct turns one giant, opaque mapping
				$c_t​⟼a_t$
	Into a sequence of smaller adaptive pieces.
* By the time for selecting $a_t$ the information needed has already been surfaced in plain language, so the final selection is much cheaper.
* Splitting the decision process into atomic thoughts also allows for more direct analysis and interpretability of the agent's decision making process, rather than having to probe a massive hidden state at once.
* Zero-feedback thoughts carry an opportunity cost (no new observational data), so during finetuning, the policy learns to trade off extra reasoning steps vs action steps.
* In practice, given a prompt the ReAct agent goes through a  loop of chaining together thoughts and actions, ending when having arrived at a final goal or answer.

	The agent is essentially a perception-action loop with incrementing context that alternates between *actions* which return *observations* appended to and *thinking* which produces thoughts based on the current context and applies this to the context. When the agent has reached its final answer it uses the *Finish* tool, to produce it to the user and end the loop. 
	
	*The authors of the original paper fine-tune their model on a set of $(c_t, Thought, Action)$ sequences via standard next-token likelihood, implicitly learning when to think vs act, to limit our scope we skip this step.
	
	
	


