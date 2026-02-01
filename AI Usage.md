# AI Usage 

## Overview of AI Tools Used

During this project, we used multiple AI agents for different purposes:

- **Google NotebookLM**, for literature comprehension and synthesis 
- **GitHub Copilot**, for code completion and docstring generation
- **ChatGPT** (GPT-5 - free tier), for debugging and code optimization 
- **Gemini**  (Pro - student account), for debugging and algorithm implementation
---

## Usage by Project Phase

### 1. Literature Review Phase

We used Google NotebookLM to improve our understanding of the literature, i.e. García Martín & Goldenfeld (2006) and Kitzes & Harte (2014).
Some queries examples:
```
- Can you help me defining: nested sampling, the term conspecific, octave classification?
- Can you explain me step by step the derivation of equation [20] in the supporting text?
- If I am trying to implement the first paper (García Martín & Goldenfeld, 2006), can i add then considerations from Kitzes & Harte (2014)?
```

### 2. Baseline Model Implementation

#### Issue 1: Correlation Function Implementation

We encountered some issues in the initial implementation of the branching algorithm, since it did not produce the expected power-law correlation function, needed to validate the tree generation.
We initially used ChatGPT, simply reporting our code and objective and asking for improvements:

*Query:*
```
We are trying to implement an sar model, the individuals are placed based on self similar tree generation, the algorithm of which is given in the paper under "Methods for Generation of Distributions of Individuals. Self-similar distributions.
Is the implementation correct to obtain individuals with a power law correlation function?
[pasted code]
```
*Reply:* ChatGPT focused on general debugging "check branch probabilities," "verify angular recursion") but didn't identify the core issue with our correlation function calculation.

*Follow-up query:*
```
Can you give an example of how to fix the correlation function?
```
*Reply:* ChatGPT assumed we were trying to implement a radial pair correlation function g(r), which didn't match the paper's methodology.

We switched to Gemini Pro's "thinking" mode. We first provided Gemini with simple code snippets to verify our own understanding of the implementation and guide its reasoning. 
We then referred to the main code as well as the figure from the paper we were trying to achieve, e.g. 
```
I want to implement the same logic in the following class but i am not sure if i am missing something with the list of list expression
```

Gemini was able to guide us to the correct normalized correlation function expression, that showed clear power-law behavior. 

---

#### Issue 2: Branch Length Scaling

Moving on with SAR computations, we experiences some issues in interpreteting the parameters used by García Martín & Goldenfeld (2006), in particular regarding initial branch length l0 and decay rates. 
As a result, plots were showing non-linear behavior on log-log scale, or produced z values outside the expected range [0.2, 0.4]. 
We initially asked ChatGPT to help us with parameters tuning in general, to allow for broader results.

*Query:*
```
Can you give guidance on the parameters for the Field class, to get a pronounced power law in the s_c values?
```
*Reply:* ChatGPT provided "parameter ranges that reliably give power laws" together with physics intuition, but did not provide the solution we were looking for.

As we realised that the problem was the initial branch length, we decided to shift our focus on the decay rates. First, in a new chat we asked ChatGPT about its general knowledge on bisecting tree algorithms:

*Query (New chat):*
```
Do you know any paper that implemented a self similar distribution with power law correlation function using a bisecting tree algorithm?
```
*Reply:* ChatGPT provided examples of papers and models that implement self-similar distributions with power-law correlation properties.
From there, we asked specifically about how to to length size of the branch at each time step, highlighting the issue we had about SAR exponent not be correct. Finally, ChatGPT was able to give advice on the schedule we could follow to reduce exponentially the branch size. 
*Follow-up query:*
```
Is there any other schedule for lambda rather than lambda / 2^n i can use to get to a similar result
```
*Reply:* ChatGPT provided concrete examples to implement 'scale invariance in expectation.'

### 3. EAR Model Extension

#### Issue 3: Extinction Parameter q Implementation
When trying to compute the values for $q$ (from the Extinction-Area Relationship, computing through root finding), no value was found. Thus, ChatGPT was used to check the used method and to try to find the main problem. Using ChatGPT, after a few general prompts, useful insights were found about what went wrong, which were the bounds, as the intersection points often fell outside the bounds that were set up.
As an example:

*Query:*
```
[insert formula from paper] From the above formula, how would I get q?
```
*Reply:* ChatGPT showed mathematical reasoning and clarified the reasoning needed for implementation.

---
## Conclusions on AI Usage

Throughout our project, AI was crucial to solve technical implementation challenges, but only when used strategically. Targeted queries with full context (code, objectives, what we'd already tried), breaking problems into smaller queries and engaging in iterative dialogue was far more effective than expecting immediate solutions. We tried using AI as a collaborative reasoning partner, guiding its reasoning. On the other hand, generic requests ("fix this code") or copy-pasting without context did not produce truly helpful results.

**Links to Complete Conversations**
- Tree generation: https://chatgpt.com/share/697b30f0-2c04-800f-84b8-5cae60c7d291
  - Solution: https://amsuni-my.sharepoint.com/:u:/g/personal/anna_de_martin_student_uva_nl/IQBGHOOHwBYRQ4uSB2i1eo7wAdsO1wtbHD3jJXgJD0PaZsQ?e=piht9f
- Power law: https://chatgpt.com/share/697b3092-0994-800f-8f16-b357e175fe5d
  - Solution: https://chatgpt.com/share/697b312b-c9b0-800a-9001-0bd77cdbb616
- Parameter q: https://chatgpt.com/share/697b32f5-bc74-8000-a4db-01ee48b4e9cc


