#include <gtest/gtest.h>
#include "engine.h"

using whisper_xpu::merge_segments;

TEST(MergeSegments, Empty) {
    EXPECT_EQ(merge_segments({}), "");
    EXPECT_EQ(merge_segments({""}), "");
    EXPECT_EQ(merge_segments({nullptr}), "");
    EXPECT_EQ(merge_segments({"", "", nullptr}), "");
}

TEST(MergeSegments, SingleSegment) {
    EXPECT_EQ(merge_segments({"hello world"}), "hello world");
}

TEST(MergeSegments, DistinctSegments) {
    EXPECT_EQ(merge_segments({"hello", "world"}), "hello world");
}

TEST(MergeSegments, ExactSuffixDuplicate) {
    // Second segment is exact duplicate of the end of first
    EXPECT_EQ(merge_segments({"hello world", "world"}), "hello world");
    EXPECT_EQ(merge_segments({"the cat sat", "cat sat"}), "the cat sat");
}

TEST(MergeSegments, OverlapSuffix) {
    // Partial overlap: "world" is shared
    EXPECT_EQ(merge_segments({"hello world", "world test"}), "hello world test");
    EXPECT_EQ(merge_segments({"this is a test", "a test message"}), "this is a test message");
}

TEST(MergeSegments, FullDuplicate) {
    EXPECT_EQ(merge_segments({"same text", "same text"}), "same text");
}

TEST(MergeSegments, MultipleOverlaps) {
    auto r = merge_segments({"hello world", "world test", "test message"});
    EXPECT_EQ(r, "hello world test message");
}

TEST(MergeSegments, NoOverlapShort) {
    // overlap minimum is 5 chars; "is" is too short
    EXPECT_EQ(merge_segments({"this is", "is good"}), "this is is good");
}

TEST(MergeSegments, Whitespace) {
    EXPECT_EQ(merge_segments({"hello ", " world"}), "hello  world");
}

TEST(MergeSegments, VectorStringOverload) {
    std::vector<std::string> v = {"a b c", "b c d", "c d e"};
    EXPECT_EQ(merge_segments(v), "a b c d e");
}
