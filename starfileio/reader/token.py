from enum import Enum


class TokenType(Enum):
    LITERAL = 0
    NEWLINE = 1
    COMMENT = 2
    LOOP = 3
    PROPERTY = 4


class Token:
    def __init__(self, token_type: TokenType, value, location):
        self.token_type = token_type
        self.value = value
        self.location = (location[0] + 1, location[1] + 1)

    @staticmethod
    def tokenize(text):
        """
        This function tokenizes a text into several tokens.
        :param text: a given text
        :return: list of tokens
        """
        tokens = list()

        # Split the text into several lines
        lines = text.split('\n')
        for line_number, line in enumerate(lines):

            # The first index of a non-space-or-hash sequence of characters. None means there is no sequence found
            first = None
            for index, char in enumerate(line):
                if not char.isspace() and char != '#':

                    # Set the first index of the sequence if it is None
                    if first is None:
                        first = index
                    continue
                elif first is not None:
                    # If a space or # and the sequence are found, classifies the sequence as
                    #   LOOP if it is 'loop_'
                    #   PROPERTY if it starts with '_'
                    #   LITERAL otherwise

                    if line[first] == '_':
                        tokens.append(Token(TokenType.PROPERTY, line[first:index], (line_number, first)))
                    elif line[first:index] == 'loop_':
                        tokens.append(Token(TokenType.LOOP, line[first:index], (line_number, first)))
                    else:
                        tokens.append(Token(TokenType.LITERAL, line[first:index], (line_number, first)))

                    # Set that there is no sequence found
                    first = None
                if char == '#':
                    # Anything after the # character is a comment

                    tokens.append(Token(TokenType.COMMENT, line[index + 1:].strip(), (line_number, index)))
                    break
                elif not char.isspace():
                    raise IOError(f"Got unexpected {char} at (Line {line_number}, Column {index}).")
            if first is not None:
                # Classifies the sequence if there is an end of line

                if line[first] == '_':
                    tokens.append(Token(TokenType.PROPERTY, line[first:], (line_number, first)))
                elif line[first:] == 'loop_':
                    tokens.append(Token(TokenType.LOOP, line[first:], (line_number, first)))
                else:
                    tokens.append(Token(TokenType.LITERAL, line[first:], (line_number, first)))

            # Add a NEWLINE token
            tokens.append(Token(TokenType.NEWLINE, None, (line_number, 0)))
        return tokens
