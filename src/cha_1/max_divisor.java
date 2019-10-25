package cha_1;

public class max_divisor {
    public static int gcd(int p, int q){
        if (q == 0){
            return p;
        }

        int r = p % q;

        return gcd(q, r);
    }

    public static void main(String[] args){
        int max = max_divisor.gcd(20,100);
        System.out.println(max);
    }
}
