import pandas as pd

from starfileio.reader.token import Token, TokenType


def read(path):
    """
    This function reads a starfile with a *.star extension into a tuple of a list of Pandas DataFrame, a list of Data
    Specifier, and a list of comments

    It reads the file and extracts the lists from the parsing function.

    :param path: the path to the starfile to be read
    :return: a tuples of a list of Pandas DataFrames, list of specifiers, and list of comments
    """

    with open(path, mode='r') as file:
        frames, specifiers, comments = parse(file.read())

    return frames, specifiers, comments


def parse(raw_starfile):
    """
    This function parses a starfile into a tuple of a list of Pandas DataFrame, a list of Data Specifier, and a list of
    comments.

    It tokenizes the file and if it finds a specifier, it starts parsing in the following order:
        1. Specifier
        2. Columns      (as column names)
        3. Rows         (as a Pandas Dataframe together with the Columns)

    :param raw_starfile: the starfile to be parsed
    :return: a tuples of a list of Pandas DataFrames, list of specifiers, and list of comments
    """
    tokens = Token.tokenize(raw_starfile)
    frames = []
    comments = []
    specifiers = []
    while lookahead(tokens, TokenType.LITERAL, [TokenType.NEWLINE, TokenType.COMMENT]):
        specifier_comments, specifier = parse_specifier(tokens)
        column_comments, columns = parse_columns(tokens)
        rows_comments, data = parse_rows(tokens, columns)
        comments.append(specifier_comments + column_comments + rows_comments)
        specifiers.append(specifier)
        frames.append(data)
    parse_newline_or_comments(tokens)
    if len(tokens) > 0:
        raise IOError(f"Expected a specifier or an end of token but got {tokens[0].token_type}")
    return frames, specifiers, comments


def parse_newline_or_comments(tokens):
    """
    This function takes a token queue and dequeues any NEWLINE token and COMMENT token while storing the comments from
    the COMMENT tokens.

    :param tokens: a queue of tokens
    :return: list of comments retrieves from the dequeued COMMENT tokens
    """
    comments = []
    while True:
        comment_token = check_then_consume(tokens, TokenType.COMMENT)
        if comment_token is not None:
            comments.append(comment_token.value)
        elif not check_then_consume(tokens, TokenType.NEWLINE):
            break
    return comments


def parse_specifier(tokens):
    """
    This function takes a token queue, gets comments, and consumes (matches) a specifier as a LITERAL token.

    :param tokens: a queue of tokens
    :return: a tuple of comments and the parsed specifier
    """
    comments = parse_newline_or_comments(tokens)
    specifier = consume(tokens, TokenType.LITERAL)
    return comments, specifier.value


def parse_columns(tokens):
    """
    This function takes a token queue, gets comments, consumes (matches) the "loop_" keyword as a LOOP token
    following by a NEWLINE token, and parses the column names

    :param tokens: a queue of tokens
    :return: a tuple of comments and column names
    """
    comments = parse_newline_or_comments(tokens)
    columns = []
    consume(tokens, TokenType.LOOP)
    consume(tokens, TokenType.NEWLINE)
    while check(tokens, TokenType.PROPERTY):
        column = parse_column(tokens)
        columns.append(column)
    return comments, columns


def parse_column(tokens):
    """
    This function takes a token queue, consumes a column name token as a PROPERTY token, and tries to consume
    a COMMENT token to retrieve the comment if existed.

    The PROPERTY token captures anything starting with "_", therefore the column name be the value of the token
    without the "_".

    :param tokens: a token queue
    :return: a tuple of comments and the column name
    """
    column = consume(tokens, TokenType.PROPERTY)
    check_then_consume(tokens, TokenType.COMMENT)
    consume(tokens, TokenType.NEWLINE)
    return column.value[1:]


def parse_rows(tokens, columns):
    """
    This function takes a token queue, gets comments, tries to consume LITERAL tokens as a rows which matches
    the number of columns before getting a new line, and converts the rows to a Pandas DataFrame.

    :param tokens: a queue of tokens
    :param columns: a list of column names
    :return: a tuple of comments and Pandas DataFrames
    """
    comments = parse_newline_or_comments(tokens)
    end = False
    rows = []
    while not end:
        data = []
        for i in range(len(columns)):
            token = check_then_consume(tokens, TokenType.LITERAL)
            if token is None:
                end = True
                break
            else:
                data.append(token.value)
        else:
            consume(tokens, TokenType.NEWLINE)
            rows.append(data)
    return comments, pd.DataFrame(rows, columns=columns)


def check(tokens, token_type):
    """
    This function checks if the first token from the given token queue matches a given token type.

    :param tokens: a queue of tokens
    :param token_type: a token type to be matched
    :return: a boolean value indicating the match
    """

    if len(tokens) == 0:
        raise IOError(f"Expected {token_type} but there are not enough token.")
    if tokens[0].token_type == token_type:
        return True
    return False


def consume(tokens, token_type):
    """
    This function consumes the first token from the given token queue. If the token type of the first
    token does not match the token type to be matched, this function will raise a parsing error.

    :param tokens: a queue of tokens
    :param token_type: a token type to be matched
    :return: the first token
    """
    if len(tokens) == 0:
        raise IOError(f"Expected {token_type} but there are enough token.")
    if tokens[0].token_type == token_type:
        return tokens.pop(0)
    else:
        raise IOError(f"Expected {token_type} but got {tokens[0].token_type} at {tokens[0].location}.")


def check_then_consume(tokens, token_type):
    """
    This function checks the first token from the given token queue and consumes it if matched. Otherwise,
    it returns a None

    :param tokens: a queue of tokens
    :param token_type: a token type to be matched
    :return: the first token or None
    """
    if len(tokens) > 0 and tokens[0].token_type == token_type:
        return consume(tokens, token_type)
    return None


def lookahead(tokens, token_type_target, ignores):
    """
    This function looks for a token type while ignoring token types from the ignores list

    :param tokens: a queue of tokens
    :param token_type_target: a token type to be found
    :param ignores: a list of token types to be ignored
    :return: a boolean value indicating a found token
    """
    for i in range(len(tokens)):
        if tokens[i].token_type == token_type_target:
            return True
        elif tokens[i].token_type in ignores:
            continue
        else:
            break
    return False
