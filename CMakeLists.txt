cmake_minimum_required(VERSION 3.20)
project(word2vec VERSION 1.0 LANGUAGES CXX)

add_library(w2vCompilerOps INTERFACE)

target_compile_features(w2vCompilerOps
	INTERFACE cxx_std_23
)

add_executable(word2vec
	src/word2vec.cc
	src/kv_options.cc
	src/w2v_options.cc
	src/word_reader.cc
	src/sentence_reader.cc
	src/unigram_table.cc
	src/huffman_tree.cc
	src/dictionary.cc
	src/word_layer.cc
	src/w2v_launcher.cc
)

find_package(Threads REQUIRED)

target_link_libraries(word2vec
	PUBLIC w2vCompilerOps
	PRIVATE Threads::Threads
)

target_include_directories(word2vec PUBLIC include)

target_compile_options(word2vec
	PRIVATE
		-Wall
		-Wextra
		-Wpedantic

		$<$<CONFIG:Debug>:
            -g
            -O1
            -fno-omit-frame-pointer
            -fsanitize=address,undefined
        >
)

target_link_options(word2vec
	PRIVATE $<$<CONFIG:Debug>:
            -fsanitize=address,undefined
        >
)

enable_testing()
include(CTest)

add_subdirectory(tests)
