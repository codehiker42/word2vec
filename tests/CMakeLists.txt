include(FetchContent)
FetchContent_Declare(
	googletest
	URL
	https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

function(add_gtest targetName)
	add_executable(${targetName} ${ARGN})

	target_link_libraries(${targetName}
		PUBLIC w2vCompilerOps
		PRIVATE
			GTest::gtest_main
			GTest::gmock_main
	)

	target_include_directories(${targetName} PRIVATE ../include ./)

	gtest_discover_tests(${targetName})
endfunction()

set(SRC "../src")

add_gtest(buffer_array_test buffer_array_test.cc)
add_gtest(word_reader_test word_reader_test.cc ${SRC}/word_reader.cc)
add_gtest(unigram_table_test unigram_table_test.cc ${SRC}/unigram_table.cc)
add_gtest(huffman_tree_test huffman_tree_test.cc ${SRC}/huffman_tree.cc)
add_gtest(
	dictionary_test
	dictionary_test.cc
	${SRC}/huffman_tree.cc
	${SRC}/unigram_table.cc
	${SRC}/word_reader.cc
	${SRC}/dictionary.cc
)
add_gtest(
	sentence_reader_test
	sentence_reader_test.cc
	${SRC}/sentence_reader.cc
	${SRC}/dictionary.cc
	${SRC}/word_reader.cc
	${SRC}/unigram_table.cc
	${SRC}/huffman_tree.cc
)
add_gtest(
	word_layer_test
	word_layer_test.cc
	${SRC}/word_layer.cc
	${SRC}/dictionary.cc
	${SRC}/huffman_tree.cc
	${SRC}/unigram_table.cc
)

