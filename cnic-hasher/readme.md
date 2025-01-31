# Hash Function

Simple function that left shifts the middle part of the CNIC number by 'last-bit'-th digits. This is ideal because we were granted that the first part would be constant, and we know that the middle part depends on the family number, which is effectively unique for our use-case, and the last part depends on, I believe, the gender.

# Data Structure

A simple structure comprising integers for length and capacity, and a `Vec<Option<String>>` for the actual array itself. The string variable was wrapped in an `Option` for enhanced type-safety and run-time checks.

# Collision Resolution

Simple linear probing, i.e if an index generated against an input CNIC is occupied, simply increment the index and try again, looping to the start of the array if the end is reached.

# Collision Statistics

For approximately normal distribution of random inputs (as we can safely expect in this scenario), we will have either few or easily-handled collisions; this is because the capacity of our vector doubles as we occupy 60% of the available space, ensuring that the probability of a collisions remains minimal.

A more rigorous mathematical proof is severely hampered by the fact that neither the number of elements, nor the size of the hashmap, are constants. Theoretically speaking, however, the maximum number of collisions would arise when, somehow, only the first 60% indexes are fully occupied, and we are the very cusp of crossing the load factor (60%) elements. Should a CNIC be inserted then, it would collide with every single CNIC thereby resulting in ***approximately*** 0.6 \* `capacity` collisions.
