explaining the messy folder structure:

.claude - idk but this is how claude works.
.vscode - ignore this
_pychahe - ignore this too

old - all that old code i wrote for other ppg stuff, you can ignore this

depreciated - here I put the code and files that aren't particularly old, but are for older prototypes of our work. things like taking port input for just one ppg (now outdated since we use the multiplexer to collect many), taking port input for multiple ppg but no ecg, etc.


picofix - the nuke files for the raspberry pi if it tweaks out again. the instructions on how to use it are inside the picofix folder itself as instructions.md

invalidSessions - Any session that isn't going to be considered as a true recording in our paper is put into invalidSessions. parlty for additional testing purposes, partly because im scared to delete stuff.

MDPIdata - where we will store the actual data. within it there are folders, one for each subject. for ourselves, i have our names on it so we can easily test/debug, but dont forget that all volunteer subjects must remain anonymous and use an ID.
