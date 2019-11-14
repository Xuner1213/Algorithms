package cha_1;


import java.util.*;

class Solution {
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> triangle = new ArrayList<List<Integer>>();

        if (numRows == 0) {
            return triangle;
        }

        triangle.add(new ArrayList<>());
        triangle.get(0).add(1);

        for (int rowNum = 1; rowNum < numRows; rowNum++) {
            List<Integer> row = new ArrayList<>();
            List<Integer> preRow = triangle.get(rowNum - 1);

            row.add(1);

            for (int i = 1; i < rowNum; i++) {
                row.add(preRow.get(i -1) + preRow.get(i));
            }

            row.add(1);
            triangle.add(row);
        }
        return triangle;
    }

    public List<Integer> getRow(int rowIndex) {
        List<Integer> row = new ArrayList<Integer>();
        row.add(1);
        for (int i = 1; i <= rowIndex; i++) {
            for (int j = i - 1; j > 0; j--) {
                row.set(j, row.get(j - 1) + row.get(j));
            }
            row.add(1);
        }
        return row;
    }

    public int maxProfit(int prices[]) {
        int minprice = Integer.MAX_VALUE;
        int maxprofit = 0;
        for (int i = 0; i < prices.length; i++) {
            if (prices[i] < minprice) {
                minprice = prices[i];
            } else if (prices[i] - minprice > maxprofit) {
                maxprofit = prices[i] - minprice;
            }
        }
        return maxprofit;
    }

    public int maxProfitNew(int prices[]) {
        int maxprofit = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > prices[i - 1]) {
                maxprofit += prices[i] - prices[i - 1];
            }
        }
        return maxprofit;
    }

    public boolean isPalindrome(String s) {
        int i = 0, j = s.length() - 1;
        while (i < j) {
            while (i < j && !Character.isLetterOrDigit(s.charAt(i))) {
                i++;
            }
            while (i < j && !Character.isLetterOrDigit(s.charAt(j))) {
                j --;
            }
            if (Character.toLowerCase(s.charAt(i)) != Character.toLowerCase(s.charAt(j))) {
                return false;
            }
            i++;
            j--;
        }
        return true;
    }

    public int singleNumber(int[] nums) {
        Map<Integer, Integer> hashMap = new HashMap<>();
        for (Integer i : nums) {
            Integer count = hashMap.get(i);
            count = count == null ? 1 : ++count;
            hashMap.put(i, count);
        }
        for (Integer i : hashMap.keySet()) {
            Integer count = hashMap.get(i);
            if (count == 1) {
                return i;
            }
        }
        return -1;
    }

    public int singleNumberNew(int[] nums) {
        int ans = nums[0];
        if (nums.length > 1) {
            for (int i = 1; i < nums.length; i++) {
                ans = ans ^ nums[i];
            }
        }
        return ans;
    }

    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        ListNode slow = head;
        ListNode fast = head.next;

        while (slow != fast) {
            if (fast == null || fast.next == null) {
                return false;
            }
            slow = slow.next;
            fast = fast.next.next;
        }
        return true;
    }

    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) {
            return null;
        }
        ListNode pA = headA;
        ListNode pB = headB;
        while (pA != pB) {
            pA = pA == null ? headB : pA.next;
            pB = pB == null ? headA : pB.next;
        }
        return pA;
    }

    public int[] twoSum(int[] numbers, int target) {
        int left = 0;
        int right = numbers.length - 1;
        while (left < right) {
            int sum = numbers[left] + numbers[right];
            if (sum == target) {
                return new int[]{left + 1, right + 1};
            } else if (sum > target) {
                right--;
            } else if (sum < target) {
                left++;
            }
        }
        throw new RuntimeException("在数组中没有找到这样的两个数，使得他们的和为指定值");
    }

    public String convertToTitle(int n) {
        String AZ = "#ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        char[] CZ = AZ.toCharArray();

        StringBuilder sb = new StringBuilder();
        while (n > 0) {
            if (n % 26 == 0) {
                sb.append('Z');
                n = n / 26 - 1;
            } else {
                sb.append(CZ[n % 26]);
                n = n / 26;
            }
        }
        return sb.reverse().toString();
    }

    public int majorityElement(int[] nums) {
        Arrays.sort(nums);
        return nums[nums.length / 2];
    }

    public int titleToNumber(String s) {
        int ans = 0;
        for (int i = 0; i < s.length(); i++) {
            int num = s.charAt(i) - 'A' + 1;
            ans = ans * 26 + num;
        }
        return ans;
    }

    public int trailingZeroes(int n) {
        return n == 0 ? 0 : n / 5 + trailingZeroes(n / 5);
    }

    public void rotate(int[] nums, int k) {
        k %= nums.length;
        reverse(nums, 0, nums.length - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.length - 1);
    }

    public void reverse(int[] nums, int start, int end) {
        while(start < end) {
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start++;
            end--;
        }
    }

    // you need treat n as an unsigned value
    public int reverseBits(int n) {
        //00000010100101000001111010011100
        //00000010100101000001111010011100
        int result = 0;
        for (int i = 0; i <= 32; i++) {
            int tmp = n >> i;
            tmp = tmp & 1;
            tmp = tmp << (31 - i);
            result |= tmp;
        }
        return result;
    }

    public int hammingWeight(int n) {
        int bits = 0;
        int mask = 1;
        for (int i = 0; i < 32; i++) {
            if ((n & mask) != 0) {
                bits++;
            }
            n = n >> 1;
        }
        return bits;
    }

    public int rob(int[] nums) {
        int preMax = 0;
        int curMax = 0;
        for (int x : nums) {
            int tmp = curMax;
            curMax = Math.max(preMax + x, curMax);
            preMax = tmp;
        }
        return curMax;
    }

    public boolean isHappy(int n) {
        Set<Integer> set = new HashSet<Integer>();
        int m = 0;
        while (true) {
            while (n != 0) {
                m += Math.pow(n % 10, 2);
                n /= 10;
            }
            if (m == 1) {
                return true;
            }
            if (set.contains(m)) {
                return false;
            } else {
                set.add(m);
                n = m;
                m = 0;
            }

        }
    }

    public ListNode removeElements(ListNode head, int val) {
        if (head == null) {
            return null;
        }
        head.next = removeElements(head.next, val);
        if (head.val == val) {
            return head.next;
        } else {
            return head;
        }
    }

    public int countPrimes(int n) {
        int[] nums = new int[n];
        for (int i = 0; i < n; i++) {
            nums[i] = 1;
        }
        for (int i = 2; i < n ; i++) {
            if (nums[i] == 1) {
                for (int j = 2; i * j < n; j++) {
                    nums[i * j] = 0;
                }
            }
        }
        int res = 0;
        for (int i = 2; i < n; i++) {
            if (nums[i] == 1) {
                res++;
            }
        }
        return res;
    }

    public boolean isIsomorphic(String s, String t) {
        Map<Character, Character> map = new HashMap<Character, Character>();
        if (s.length() != t.length()) {
            return false;
        }
        for (int i = 0; i < s.length(); i++) {
            char ss = s.charAt(i);
            char tt = t.charAt(i);
            if (map.containsKey(ss)) {
                if (map.get(ss) != tt) {
                    return false;
                }
            } else {
                if (map.containsValue(tt)) {
                    return false;
                }
                map.put(ss, tt);
            }
        }
        return true;
    }

    public ListNode reverseList(ListNode head) {
        ListNode pre = null;
        ListNode cur = head;
        while (cur != null) {
            ListNode nextTmp = cur.next;
            cur.next = pre;
            pre = cur;
            cur = nextTmp;
        }
        return pre;
    }

    public ListNode reverseList1(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode p = reverseList1(head.next);
        head.next.next = head;
        head.next = null;
        return p;
    }

    public boolean containsDuplicate(int[] nums) {
        Set<Integer> set = new HashSet<Integer>(nums.length);
        for (int num : nums) {
            if (set.contains(num)) {
                return true;
            } else {
                set.add(num);
            }
        }
        return false;
    }

    public boolean containsDuplicate1(int[] nums) {
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 1; i++) {
            if (nums[i] == nums[i + 1]) {
                return true;
            }
        }
        return false;
    }

    public boolean containsNearbyDuplicate(int[] nums, int k) {
        Set<Integer> set = new HashSet<Integer>();
        for (int i = 0; i < nums.length; i++) {
            if (set.contains(nums[i])) {
                return true;
            }
            set.add(nums[i]);
            if (set.size() > k) {
                set.remove(nums[i - k]);
            }
        }
        return false;
    }

    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        TreeNode right = invertTree(root.right);
        TreeNode left = invertTree(root.left);
        root.left = right;
        root.right = left;
        return root;
    }

    public boolean isPowerOfTwo(int n) {
        if (n <= 0) {
            return false;
        }
        return (n & (n - 1)) == 0;
    }

    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) {
            return true;
        }

        int len = 0;
        for (ListNode cur = head; cur != null; cur = cur.next) {
            len++;
        }

        ListNode pre = null;
        ListNode cur = head;
        for (int i = 0; i < len / 2; i++) {
            ListNode tmp = cur.next;
            cur.next = pre;
            pre = cur;
            cur = tmp;
        }
        if ((len & 1) == 1) {
            cur = cur.next;
        }

        for (ListNode p = cur, q = pre; p != null && q != null; p = p.next, q = q.next) {
            if (p.val != q.val) {
                return false;
            }
        }
        return true;
    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        int pval = p.val;
        int qval = q.val;
        TreeNode node = root;
        while (node != null) {
            int parentVal = node.val;
            if (parentVal < pval && parentVal < qval) {
                node = node.right;
            } else if (parentVal > pval && parentVal > qval) {
                node = node.left;
            } else {
                return node;
            }
        }
        return null;
    }

    public void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }

    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        }
        int[] counter = new int[26];
        for (int i = 0; i < s.length(); i++) {
            counter[s.charAt(i) - 'a']++;
            counter[t.charAt(i) - 'a']--;
        }
        for (int count : counter) {
            if (count != 0) {
                return false;
            }
        }
        return true;
    }

    public void construct_paths(TreeNode root, String path, LinkedList<String> paths) {
        if (root != null) {
            path += Integer.toString(root.val);
            if (root.left == null && root.right == null) {
                paths.add(path);
            } else {
                path += "->";
                construct_paths(root.left, path, paths);
                construct_paths(root.right, path, paths);
            }
        }
    }

    public int addDigits(int num) {
        return (num - 1) % 9 + 1;
    }

    public List<String> binaryTreePaths(TreeNode root) {
        LinkedList<String> paths = new LinkedList<String>();
        construct_paths(root, "", paths);
        return paths;
    }

    public boolean isUgly(int num) {
        if (num < 0) {
            return false;
        }

        while (num % 2 == 0 || num % 3 == 0 || num % 5 == 0) {
            if (num % 2 == 0) {
                num /= 2;
            } else if (num % 3 == 0) {
                num /= 3;
            } else if (num % 5 == 0) {
                num /= 5;
            }
        }
        return num == 1 ? true : false;
    }

    public int missingNumber(int[] nums) {
        Set<Integer> set = new HashSet<Integer>();
        for (int num : nums) {
            set.add(num);
        }
        int exceptedNum = nums.length + 1;
        for (int i = 0; i < exceptedNum; i++) {
            if (!set.contains(i)) {
                return i;
            }
        }
        return -1;
    }

    public void moveZeroes(int[] nums) {
        int k = 0;
        int len = nums.length;
        for (int i = 0; i < len; i++) {
            if (nums[i] != 0) {
                swapArrays(nums, i, k++);
            }
        }
    }

    private void swapArrays(int[] nums,int first,int second){
        if (first == second) {
            return;
        }
        int temp = nums[first];
        nums[first] = nums[second];
        nums[second] = temp;
    }

    public boolean wordPattern(String pattern, String str) {
        String[] s = str.split(" ");
        if (s.length != pattern.length()) {
            return false;
        }
        Map<Character, String> map = new HashMap<Character, String>();
        for (int i = 0; i < s.length; i++) {
            if (!map.containsKey(pattern.charAt(i))) {
                if (map.containsValue(s[i])) {
                    return false;
                }
                map.put(pattern.charAt(i), s[i]);
            } else {
                if (!map.get(pattern.charAt(i)).equals(s[i])) {
                    return false;
                }
            }
        }
        return true;
    }

    public boolean canWinNim(int n) {
        return (n % 4 != 0);
    }

    public boolean isPowerOfThree(int n) {
        if (n < 1) {
            return false;
        }
        while (n % 3 == 0) {
            n /= 3;
        }
        return n == 1;
    }

    public boolean isPowerOfFour(int num) {
        if (num < 0) {
            return false;
        }

        if ( (num&(num-1)) != 0) {
            return false;
        }

        if ((num & 0x55555555) == num) {
            return true;
        }
        return false;
    }

    public void reverseString(char[] s) {
        if (s == null || s.length < 2) {
            return;
        }

        int pre = -1;
        int last = s.length;
        while (++pre < --last) {
            char tmp = s[pre];
            s[pre] = s[last];
            s[last] = tmp;
        }
    }

    public String reverseVowels(String s) {
        List vowels = new ArrayList();
        StringBuilder stringBuilder = new StringBuilder(s);
        int len = s.length();
        int head = 0;
        int tail = len - 1;
        char temp;
        vowels.add('a');
        vowels.add('e');
        vowels.add('i');
        vowels.add('o');
        vowels.add('u');
        vowels.add('A');
        vowels.add('E');
        vowels.add('I');
        vowels.add('O');
        vowels.add('U');
        while (head < tail) {
            if (vowels.contains(s.charAt(head)) == true && vowels.contains(s.charAt(tail)) == true) {
                temp = s.charAt(head);
                stringBuilder.setCharAt(head, s.charAt(tail));
                stringBuilder.setCharAt(tail, temp);
                head++;
                tail--;
            } else if (vowels.contains(s.charAt(head)) == true && vowels.contains(s.charAt(tail)) != true) {
                tail--;
            } else if (vowels.contains(s.charAt(head)) != true && vowels.contains(s.charAt(tail)) == true) {
                head++;
            } else {
                head++;
                tail--;
            }
        }
        return stringBuilder.toString();
    }

    public List<Integer> topKFrequent(int[] nums, int k) {
        HashMap<Integer, Integer> count = new HashMap();
        for (int n : nums) {
            count.put(n, count.getOrDefault(n, 0) + 1);
        }

        PriorityQueue<Integer> heap = new PriorityQueue<>((n1, n2) -> count.get(n1) - count.get(n2));

        for (int n : count.keySet()) {
            heap.add(n);
            if (heap.size() > k) {
                heap.poll();
            }
        }
        List<Integer> list = new LinkedList<>();
        while (!heap.isEmpty()) {
            list.add(heap.poll());
        }
        Collections.reverse(list);
        return list;
    }

    public int[] intersection(HashSet<Integer> set1, HashSet<Integer> set2) {
        int[] output = new int[set1.size()];
        int index = 0;
        for (Integer s : set1) {
            if (set2.contains(s)) {
                output[index++] = s;
            }
        }
        return Arrays.copyOf(output, index);
    }

    public int[] intersection(int[] nums1, int[] nums2) {
        HashSet<Integer> set1 = new HashSet<Integer>();
        for (Integer num : nums1) {
            set1.add(num);
        }
        HashSet<Integer> set2 = new HashSet<Integer>();
        for (Integer num : nums2) {
            set2.add(num);
        }
        if (set1.size() < set2.size()) {
            return intersection(set1, set2);
        } else {
            return intersection(set2, set1);
        }
    }

    public int[] intersect(int[] nums1, int[] nums2) {
        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
        List<Integer> list = new ArrayList<>();
        for (int num : nums1) {
            if (!map.containsKey(num)) {
                map.put(num, 1);
            } else {
                map.put(num, map.get(num) + 1);
            }
        }

        for (int num : nums2) {
            if (map.containsKey(num)) {
                map.put(num, map.get(num) - 1);
                if (map.get(num) == 0) {
                    map.remove(num);
                }
                list.add(num);
            }
        }

        int[] res = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            res[i] = list.get(i);
        }
        return res;
    }

    public boolean isPerfectSquare(int num){
        int i = 1;
        while (num > 0) {
            num -= i;
            i += 2;
        }
        return num == 0;
    }

    public static int getSum(int a, int b) {
        while (b != 0) {
            int res = (a & b) << 1;
            a = a ^ b;
            b = res;
        }
        return a;
    }

    public char findTheDifference(String s, String t) {
        char ans  = t.charAt(t.length() - 1);
        for (int i = 0; i < s.length(); i++) {
            ans ^= s.charAt(i);
            ans ^= t.charAt(i);
        }
        return ans;
    }

    public boolean isSubsequence(String s, String t) {
        int index = 0, i = 0;
        while (index < s.length() && t.indexOf(s.charAt(index), i) >= i) {
            i = t.indexOf(s.charAt(index), i) + 1;
            index++;
        }
        return index == s.length();
    }

    public List<String> readBinaryWatch(int num) {
        List<String> list = new ArrayList<>();
        for (int i = 0; i < 12; i++) {
            for (int j = 0; j < 60; j ++) {
                if (Integer.bitCount(i) + Integer.bitCount(j) == num) {
                    list.add(String.format("%d:%02d", i, j));
                }
            }
        }
        return list;
    }

//    private List<String> res = new ArrayList<>();
//    public List<String> readBinaryWatch1(int num) {
//        dfs(num, 0, new int[10]);
//        return res;
//    }
//
//    public void dfs(int num, int cur, int[] book) {
//        if (num == 0) {
//            int hour = book[0] + 2 * book[1] + 4 * book[2] + 8 * book[3];
//            int minute = book[4] + 2 * book[5] + 4 * book[6] + 8 * book[7] + 16 * book[8] + 32 * book[9];
//            if (hour < 12 && minute < 60) {
//                res.add(String.format("%d:%02d", hour, minute));
//            }
//            return;
//        }
//
//        for (int i = cur; i < book.length; i++) {
//            book[i] = 1;
//            dfs(num - 1, i + 1, book);
//            book[i] = 0;
//        }
//    }

    private boolean isLeftLeave(TreeNode treeNode) {
        return treeNode != null && (treeNode.left == null && treeNode.right == null);
    }

    public int sumOfLeftLeaves(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int sum = 0;
        if (isLeftLeave(root.left)) {
            sum += root.left.val;
        } else {
            sum += sumOfLeftLeaves(root.left);
        }
        sum += sumOfLeftLeaves(root.right);
        return sum;
    }

    public static String toHex(int num) {
        char[] hex = "0123456789abcdef".toCharArray();
        String s = new String();
        while (num!= 0) {
            int end = num & 15;
            s = hex[end] + s;
            num >>>= 4;
        }
        if (s.length() == 0) {
            s = "0";
        }
        return s;
    }

    public int longestPalindrome(String s) {
        int[] count = new int[128];
        for (char c : s.toCharArray()) {
            count[c]++;
        }

        int ans = 0;
        for (int v : count) {
            ans += v / 2 * 2;
            if(v % 2 == 1 && ans % 2 == 0) {
                ans++;
            }
        }
        return ans;
    }

    public List<String> fizzBuzz(int n) {
        List<String> ans = new ArrayList<String>();

        HashMap<Integer, String> hashMap = new HashMap<Integer, String>() {
            {
                put(3, "Fizz");
                put(5, "BUzz");
            }
        };

        for (int num = 1; num <= n; num++) {
            String numAnsStr = "";
            for (Integer key : hashMap.keySet()) {
                if (num % key == 0) {
                    numAnsStr += hashMap.get(key);
                }
            }
            if (numAnsStr.equals("")) {
                numAnsStr += Integer.toString(num);
            }
            ans.add(numAnsStr);
        }
        return ans;
    }

//    public int thirdMax(int[] nums) {
//        if (nums == null || nums.length == 0) {
//            throw new RuntimeException("error");
//        }
//
//        TreeSet<Integer> treeSet = new TreeSet<Integer>();
//
//        for (int elem : nums) {
//            treeSet.add(elem);
//            if (treeSet.size() > 3) {
//                treeSet.remove(treeSet.first());
//            }
//        }
//
//        return treeSet.size() < 3 ? treeSet.last() : treeSet.first();
//    }

    private long MIN = Long.MIN_VALUE;

    public int thirdMax(int[] nums) {
        if (nums == null || nums.length == 0) {
            throw new RuntimeException("nums is null or length of 0");
        }

        int n = nums.length;

        int one = nums[0];
        long two = MIN;
        long three = MIN;

        for (int i = 1; i < n; i++) {
            int now = nums[i];
            if (now == one || now == two || now == three) {
                continue;
            }

            if (now > one) {
                three = two;
                two = one;
                one = now;
            } else if (now > two) {
                three = two;
                two = now;
            } else if (now > three) {
                three = now;
            }
        }

        if (three == MIN) {
            return one;
        } else {
            return (int) three;
        }
    }

    public String addStrings(String num1, String num2) {
        StringBuilder stringBuilder = new StringBuilder("");
        int i = num1.length() - 1, j = num2.length() - 1, carry = 0;
        while (i >= 0 || j >= 0) {
            int n1 = i >= 0 ? num1.charAt(i) - '0' : 0;
            int n2 = j >= 0 ? num2.charAt(j) - '0' : 0;

            int tmp = n1 + n2 + carry;
            carry = tmp / 10;
            stringBuilder.append(tmp % 10);
            i--;
            j--;
        }
        if (carry == 1){
            stringBuilder.append(1);
        }
        return stringBuilder.reverse().toString();
    }


//    public List<List<Integer>> levelOrder(Node root) {
//        List<List<Integer>> res = new ArrayList<List<Integer>>();
//        if (root == null) {
//            return res;
//        }
//
//        Queue<Node> queue = new LinkedList<Node>();
//        queue.add(root);
//        while (!queue.isEmpty()) {
//            int count = queue.size();
//            List<Integer> list = new ArrayList<Integer>();
//            while (count-- > 0) {
//                Node cur = queue.poll();
//                list.add(cur.val);
//                for (Node node : cur.children) {
//                    if (node != null) {
//                        queue.add(node);
//                    }
//                }
//            }
//            res.add(list);
//        }
//
//        return res;
//    }

    public List<List<Integer>> levelOrder(Node root) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        if (root == null) {
            return res;
        }
        helper(root, 0, res);
        return res;
    }

    private void helper(Node root, int depth, List<List<Integer>> res) {
        if (root == null) {
            return;
        }

        if (depth + 1 > res.size()) {
            res.add(new ArrayList<Integer>());
        }

        res.get(depth).add(root.val);

        for (Node node : root.children) {
            if (node != null) {
                helper(node, depth + 1, res);
            }
        }
    }

    public int countSegments(String s) {
        int segmentCount = 0;

        for (int i = 0; i < s.length(); i++) {
            if ((i == 0 || s.charAt(i - 1) == ' ') && s.charAt(i) != ' ') {
                segmentCount++;
            }
        }

        return segmentCount;
    }

    public int pathSum(TreeNode root, int sum) {
        if (root == null) {
            return 0;
        }

        return paths(root, sum) + pathSum(root.left, sum) + pathSum(root.right, sum);
    }

    private int paths(TreeNode root, int sum) {
        if (root == null) {
            return 0;
        }

        int res = 0;

        if (root.val == sum) {
            res += 1;
        }

        res += paths(root.left, sum - root.val);
        res += paths(root.right, sum - root.val);

        return res;
    }

    public int compress(char[] chars) {
        int anchor = 0, write = 0;
        for (int read = 0; read < chars.length; read++) {
            if (read + 1 == chars.length || chars[read + 1] != chars[read]) {
                chars[write++] = chars[anchor];
                if (read > anchor) {
                    for (char c : ("" + (read - anchor + 1)).toCharArray()) {
                        chars[write++] = c;
                    }
                }
                anchor = read + 1;
            }
        }
        return write;
    }

    public int numberOfBoomerangs(int[][] points) {
        int res = 0;
        HashMap<Integer, Integer> hashMap = new HashMap<>();
        for (int i = 0; i < points.length; i++) {
            hashMap.clear();
            for (int j = 0; j < points.length; j++) {
                if (j == i) {
                    continue;
                }
                int d = (points[i][0] - points[j][0]) * (points[i][0] - points[j][0]) + (points[i][1] - points[j][1]) * (points[i][1] - points[j][1]);
                if (hashMap.containsKey(d)) {
                    res += hashMap.get(d) * 2;
                    hashMap.put(d, hashMap.get(d) + 1);
                } else {
                    hashMap.put(d, 1);
                }
            }
        }
        return res;
    }

    public void swap(int[] nums, int index1, int index2){
        if (index1 != index2) {
            nums[index1] = nums[index1] ^ nums[index2];
            nums[index2] = nums[index1] ^ nums[index2];
            nums[index1] = nums[index1] ^ nums[index2];
        }
    }

    public List<Integer> findDisappearedNumbers(int[] nums) {
        int l = nums.length;
        List<Integer> ret = new ArrayList<>();

        for (int i = 0; i < l; i++) {
            while (nums[i] != nums[nums[i] - 1]) {
                swap(nums, i, nums[i] - 1);
            }
        }

        for (int i = 0; i < l; i++) {
            if (nums[i] != i + 1) {
                ret.add(i + 1);
            }
        }
        return ret;
    }

    public int minMoves(int[] nums) {
        int moves = 0, min = Integer.MAX_VALUE;
        for (int i = 0; i < nums.length; i++) {
            moves += nums[i];
            min = Math.min(min, nums[i]);
        }
        return moves - min * nums.length;
    }

    public int findContentChildren(int[] g, int[] s) {
        if (g == null || s == null) {
            return 0;
        }
        Arrays.sort(g);
        Arrays.sort(s);
        int gi = 0, si = 0;
        while (gi < g.length && si < s.length) {
            if (g[gi] <= s[si]) {
                gi++;
            }
            si++;
        }
        return gi;
    }

    public boolean repeatedSubstringPattern(String s) {
        String str = s + s;
        return str.substring(1, str.length() - 1).contains(s);
    }

    public int hammingDistance(int x, int y) {
        return Integer.bitCount(x ^ y);
    }

    public int islandPerimeter(int[][] grid) {
        int sum = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    int lines = 4;
                    if (i > 0 && grid[i - 1][j] == 1) {
                        lines--;
                    }
                    if (i < grid.length - 1 && grid[i + 1][j] == 1) {
                        lines--;
                    }
                    if (j > 0 && grid[i][j - 1] == 1) {
                        lines--;
                    }
                    if (j < grid[0].length - 1&& grid[i][j + 1] == 1) {
                        lines--;
                    }
                    sum += lines;
                }
            }
        }
        return sum;
    }

    public static int findRadius(int[] houses, int[] heaters) {
        Arrays.sort(houses);
        Arrays.sort(heaters);
        int max = -1;
        int j = 0;
        for (int i = 0; i < houses.length; i++) {
            if (j + 1 < heaters.length && (Math.abs(houses[i] - heaters[j]) >= Math.abs(houses[i] - heaters[j + 1]))) {
                j++;
                i--;
            } else {
                if (max < Math.abs(houses[i] - heaters[j])) {
                    max = Math.abs(houses[i] - heaters[j]);
                }
            }
        }
        return max;
    }

    public static int findComplement(int num) {
        int tmp = num;
        int num2 = 1;
        while (tmp > 0) {
            tmp >>= 1;
            num2 <<= 1;
        }
        num2 -= 1;
        return num^num2;
    }

    public String[] findWords(String[] words) {
        if (words == null) {
            return null;
        }

        List<String> ans = new ArrayList<String>();
        String[] lines = new String[] {
                "qwrtyuiop",
                "asdfghjkl",
                "zxcvbnm"
        };

        for (String word : words) {
            if (judge(word, lines)) {
                ans.add(word);
            }
        }
        return ans.toArray(new String[ans.size()]);
    }

    private boolean judge(String word,String[] lines) {
        boolean ok = true;
        String find = null;

        for (String line : lines) {
            if (line.indexOf(word.charAt(0)) > -1) {
                find = line;
                break;
            }
        }

        if (find == null) {
            ok = false;
            return ok;
        }

        for (int i = 1; i < word.length(); i++) {
            if(find.indexOf(word.charAt(i)) < 0) {
                ok = false;
                break;
            }
        }
        return ok;
    }

    public int arrangeCoins(int n) {
        return (int)(Math.sqrt(2) * Math.sqrt(0.125 + n) - 0.5);
    }

    private int curcount = 1;
    private int maxcount = 1;
    TreeNode preNode = null;

    public int[] findMode(TreeNode root) {
        ArrayList<Integer> maxNumList = new ArrayList<Integer>();
        inOrder(root, maxNumList);
        int index = 0;
        int[] res = new int[maxNumList.size()];
        for (int num : maxNumList) {
            res[index++] = num;
        }
        return res;
    }

    public void inOrder(TreeNode node,List<Integer> nums){
        if (node == null) {
            return;
        }

        inOrder(node.left, nums);

        if (preNode != null) {
            if (preNode.val == node.val) {
                curcount++;
            } else {
                curcount = 1;
            }
        }

        if (curcount > maxcount) {
            maxcount = curcount;
            nums.clear();
            nums.add(node.val);
        } else if (curcount == maxcount) {
            nums.add(node.val);
        }

        inOrder(node.right, nums);
    }

    public String convertToBase7(int num) {
        StringBuilder stringBuilder = new StringBuilder();
        int flag = num >= 0 ? 1 : -1;
        num = Math.abs(num);

        if (num == 0) {
            stringBuilder.append("0");
        }

        while (num != 0) {
            stringBuilder.append(num % 7);
            num /= 7;
        }

        if (flag < 0) {
            stringBuilder.append("-");
        }

        return stringBuilder.reverse().toString();
    }

    public String[] findRelativeRanks(int[] nums) {
        int len = nums.length;

        Integer[] copy = new Integer[len];

        for (int i= 0; i < len; i++) {
            copy[i] = nums[i];
        }

        Arrays.sort(copy, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o2 - o1;
            }
        });

        Map<Integer, Integer> map = new HashMap<Integer, Integer>();

        for (int i = 0; i < len; i++) {
            map.put(copy[i], i + 1);
        }

        String[] res = new String[len];

        for (int i = 0; i < len; i++) {
            int order = map.get(nums[i]);
            String s;
            if (order == 1) {
                s = "Gold Medal";
            } else if (order == 2) {
                s = "Silver Medal";
            } else if (order == 3) {
                s = "Bronze Medal";
            } else {
                s = String.valueOf(nums[i]);
            }
            res[i] = s;
        }
        return res;
    }

    public boolean checkPerfectNumber(int num) {
        if (num == 0) {
            return false;
        }

        int sum = 1;
        int i = 2;
        double sqrt = Math.sqrt(num);

        for (; i < sqrt; i++) {
            if (num % i == 0) {
                sum += i;
                sum += num / i;
            }
        }

        if (i * i == num) {
            sum += i;
        }

        return sum == num;
    }


    public boolean checkPerfectNumbernew(int num) {
        if (num % 2 != 0) {
            return false;
        }

        int sum = 1;

        for (int i = 2; i < num / i; i++) {
            if (num % i == 0) {
                sum += i + num / i;
            }
        }
        return num == sum;
    }

    public int fib(int N) {
        int cur = 0, next = 1;
        while (N-- > 0) {
            next = cur + next;
            cur = next - cur;
        }
        return cur;
    }

    public boolean detectCapitalUse(String word) {
        char[] ch = word.toCharArray();
        int end = ch.length - 1;
        //最后一个字母为大写字母
        if (ch[end] - 'Z' <= 0) {
            for (int i = 0; i < ch.length; i++) {
                //前面有小写字母
                if (ch[i] - 'Z' > 0) {
                    return false;
                }
            }
            //最后一个字母为小写字母
        } else {
            for (int i = ch.length - 1; i >= 1; i--) {
                if (ch[i] - 'Z' <= 0) {
                    return false;
                }
            }
        }

        return true;

    }

    public int findLUSlength(String a, String b) {
        if (a.equals(b)) {
            return -1;
        }

        return a.length() > b.length() ? a.length() : b.length();
    }

    TreeNode pre = null;
    int res = Integer.MAX_VALUE;

    public int getMinimumDifference(TreeNode root) {
        inOrder(root);
        return res;
    }

    public void inOrder(TreeNode root) {
        if (root == null) {
            return;
        }

        inOrder(root.left);

        if (pre != null) {
            res = Math.min(res, root.val - pre.val);
        }

        pre = root;
        inOrder(root.right);
    }

    public int findPairs(int[] nums, int k) {
        if (nums == null || nums.length < 2 || k < 0) {
            return 0;
        }

        HashMap<Integer, Integer> hashMap = new HashMap<Integer, Integer>();

        for (int num : nums) {
            hashMap.put(num, hashMap.getOrDefault(num, 0) + 1);
        }

        int count = 0;
        Set<Integer> set = hashMap.keySet();

        for (int key : set) {
            int sub = k + key;
            if (sub == key) {
                count += (hashMap.get(key) >= 2 ? 1 : 0);
            } else if (hashMap.containsKey(sub)){
                count += 1;
            }
        }

        return count;
    }

    private int sum = 0;
    public TreeNode convertBST(TreeNode root) {
        if (root != null) {
            convertBST(root.right);
            sum += root.val;
            root.val = sum;
            convertBST(root.left);
        }
        return root;
    }

    public String reverseStr(String s, int k) {
        char[] a = s.toCharArray();
        for (int start = 0; start < a.length; start += 2 * k) {
            int i = start, j = Math.min(start + k - 1, a.length - 1);
            while (i < j) {
                char temp = a[i];
                a[i++] = a[j];
                a[j--] = temp;
            }
        }
        return new String(a);
    }

    int ans;

    public int depth(TreeNode node) {
        if (node == null) {
            return 0;
        }

        int L = depth(node.left);
        int R = depth(node.right);
        ans = Math.max(ans, L + R + 1);
        return Math.max(L, R) + 1;
    }

    public int diameterOfBinaryTree(TreeNode root) {
        ans = 1;
        depth(root);
        return ans - 1;
    }

    public boolean checkRecord(String s) {
        int countA = 0;
        for(int i = 0; i < s.length() && countA < 2; i++) {
            if (s.charAt(i) == 'A') {
                countA++;
            }
            if (i <= s.length() - 3 && s.charAt(i) == 'L' && s.charAt(i + 1) == 'L' && s.charAt(i + 2) == 'L') {
                return false;
            }
        }
        return countA < 2;
    }

    public String reverse(String s) {
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            stringBuilder.insert(0, s.charAt(i));
        }
        return stringBuilder.toString();
    }

    public String reverseWords(String s) {
        StringBuilder result = new StringBuilder();
        StringBuilder word = new StringBuilder();

        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) != ' ') {
                word.append(s.charAt(i));
            } else {
                result.append(reverse(word.toString()));
                result.append(' ');
                word.setLength(0);
            }
        }
        result.append(reverse(word.toString()));
        return result.toString();
    }


    public static void main(String[] args) {
        int a = 5;
        System.out.println(findComplement(a));
    }
}

class ListNode {
    int val;
    ListNode next;
    ListNode(int x) {
        val = x;
        next = null;
    }
 }

class MinStack {

    private Stack<Integer> data;
    private Stack<Integer> helper;

    /** initialize your data structure here. */
    public MinStack() {
        data = new Stack<Integer>();
        helper = new Stack<Integer>();
    }

    public void push(int x) {
        data.add(x);
        if (helper.isEmpty() || helper.peek() >= x) {
            helper.add(x);
        } else {
            helper.add(helper.peek());
        }
    }

    public void pop() {
        if (!data.isEmpty()) {
            data.pop();
            helper.pop();
        }
    }

    public int top() {
        if (!data.isEmpty()) {
            return data.peek();
        }
        throw new RuntimeException("栈中元素为空，此操作非法");
    }

    public int getMin() {
        if (!helper.isEmpty()) {
            return helper.peek();
        }
        throw new RuntimeException("栈中元素为空，此操作非法");
    }

    public boolean canConstruct(String ransomNote, String magazine) {
        int[] buckets = new int[26];
        for (int i = 0; i < magazine.length(); i++) {
            buckets[magazine.charAt(i) - 'a']++;
        }
        for (int i = 0; i < ransomNote.length(); i++) {
            if (--buckets[ransomNote.charAt(i) - 'a'] < 0) {
                return false;
            }
        }
        return true;
    }

    public int firstUniqChar(String s) {
        HashMap<Character, Integer> hashMap = new HashMap<Character, Integer>();
        int n = s.length();
        for (int i = 0; i < n; i++) {
            hashMap.put(s.charAt(i), hashMap.getOrDefault(s.charAt(i),0) + 1);
        }

        for (int i = 0; i < n; i++) {
            if (hashMap.get(s.charAt(i)) == 1) {
                return i;
            }
        }
        return -1;
    }
}

class MyStack {
    Deque<Integer> deque;
    /** Initialize your data structure here. */
    public MyStack() {
        deque = new LinkedList<>();
    }

    /** Push element x onto stack. */
    public void push(int x) {
        deque.add(x);
    }

    /** Removes the element on top of the stack and returns that element. */
    public int pop() {
        int last = deque.getLast();
        deque.removeLast();
        return last;
    }

    /** Get the top element. */
    public int top() {
        return deque.getLast();
    }

    /** Returns whether the stack is empty. */
    public boolean empty() {
        return deque.isEmpty();
    }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode(int x) {
        this.val = x;
    }
}

class MyQueue {
    Stack<Integer> stackPush;
    Stack<Integer> stackPop;
    /** Initialize your data structure here. */
    public MyQueue() {
        stackPush = new Stack<Integer>();
        stackPop = new Stack<Integer>();
    }

    /** Push element x to the back of queue. */
    public void push(int x) {
        stackPush.push(x);
    }

    /** Removes the element from in front of queue and returns that element. */
    public int pop() {
        if (stackPush.isEmpty() && stackPop.isEmpty()) {
            throw new RuntimeException("Queue is empty");
        }
        if (stackPop.isEmpty()) {
            while (!stackPush.isEmpty()) {
                stackPop.push(stackPush.pop());
            }
        }
        return stackPop.pop();
    }

    /** Get the front element. */
    public int peek() {
        if (stackPush.isEmpty() && stackPop.isEmpty()) {
            throw new RuntimeException("Queue is empty");
        }
        if (stackPop.isEmpty()) {
            while (!stackPush.isEmpty()) {
                stackPop.push(stackPush.pop());
            }
        }
        return stackPop.peek();
    }

    /** Returns whether the queue is empty. */
    public boolean empty() {
        return stackPush.isEmpty() && stackPop.isEmpty();
    }
}

class Node {
    public int val;
    public List<Node> children;

    public Node() {}

    public Node(int _val,List<Node> _children) {
        val = _val;
        children = _children;
    }
};



