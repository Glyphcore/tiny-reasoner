# arith (e-b-ortho-balanced-L12) ok=False

**user:** I am packing boxes. Each box holds 6 jars and I have 4 boxes. How many jars fit in total? Answer briefly.

**assistant:** Each box holds 6 jars, and there are 4 boxes.  
Total jars = $ 6 \times 4 = 24 $ jars.

_check: ok_

**user:** I actually have 5 boxes, not 4. What is the total now?

**assistant:** Total jars = $ 6 \times 5 = 30 $ jars.

_check: ok_

**user:** Good. Now explain in one sentence how you got that number.

**assistant:** I got 30 jars by multiplying the number of boxes by the number of jars each box holds.

_check: FAIL missing /5|five|6|six/_
