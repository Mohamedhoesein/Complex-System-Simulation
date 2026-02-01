# AI Usage 

## Overview of AI Tools Used

During this project, we utilized multiple AI agents for different purposes:

- **Google NotebookLM**, for literature comprehension and synthesis 
- **GitHub Copilot**, for code completion and docstring generation
- **ChatGPT** (GPT-5 - free tier), for debugging and code optimization 
- **Gemini**  (Pro - student account, for debugging and algorithm implementation
---

## Usage by Project Phase

### 1. Literature Review Phase

We used Google NotebookLM to improve our understanding of the literature, i.e. García Martín & Goldenfeld (2006) and Kitzes & Harte (2014).
Some queries examples:
- "Can you help me defining: nested sampling, the term conspecific, octave classification?"
- "Can you explain me step by step the derivation of equation [20] in the supporting text?"
- "If I am trying to implement the first paper (García Martín & Goldenfeld, 2006), can i add then considerations from Kitzes & Harte (2014)?

NotebookLM supported our learning processe with and theoretical framework. 

### 2. Baseline Model Implementation

#### Issue 1: Correlation Function Implementation

We encountered some issues in the initial implementation of the branching algorithm, since it did not produce the expected power-law correlation function, needed to validate the tree generation.
We initially used ChatGPT, simply reporting our code and objective and asking for improvements:

*Query:*
```
We are trying to implement an sar model, the individuals are placed based on self similar tree generation, the algorithm of which is given in the paper under "Methods for Generation of Distributions of Individuals. Self-similar distributions."
Is the implementation correct to obtain individuals with a power law correlation function?
[pasted code]
```
*Reply:* ChatGPT focused on general debugging ("check branching probabilities," "verify angle distributions") but didn't identify the core issue with our correlation function calculation.

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

Moving on with SAR computations, we experiences some issues in interpreteting the parameters used by García Martín & Goldenfeld (2006), in particular regarding initial branch length l₀ and decay rates. 
As a result, plots were showing non-linear behavior on log-log scale, or produced z values outside the expected range [0.2, 0.4]. 
We initially asked ChatGPT to help us with parameters...



