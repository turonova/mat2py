def write(frames, path, specifiers=None, comments=None, number_columns=True):
    if specifiers is None:
        specifiers = ["data"] * len(frames)
    if comments is None:
        comments = (None,) * len(frames)

    if len(frames) != len(specifiers) or len(frames) != len(comments) or len(specifiers) != len(comments):
        raise ValueError(f"Invalid size of the lists found. "
                         f"The sizes are (frames: {len(frames)}), "
                         f"(specifiers: {len(specifiers)}), "
                         f"and (comments: {len(comments)}).")

    with open(path, 'w') as file:
        def write_with_number(name, number):
            file.write(f"_{name} #{number}\n")

        def write_without_number(name, _):
            file.write(f"_{name}\n")

        for frame, specifier, comment in zip(frames, specifiers, comments):
            stopgap = "stopgap" in specifier
            write_function = write_without_number if not number_columns or stopgap else write_with_number
            if comment is not None:
                file.write(f"\n# {comment}\n")
            file.write(f"\n{specifier}\n\n")
            file.write("loop_\n")
            for index, column in enumerate(frame.columns):
                write_function(index, column)
            if stopgap:
                file.write("\n")
            for row in frame.itertuples(index=False):
                file.write('\t'.join(map(str, row)) + '\n')
            file.write('\n')
